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

package featuregate

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Gate enabled + field set: valid
	st.Value(&Struct{
		StringPtrField: ptr.To("Pod"),
	}).Opts([]string{"MyGate"}).ExpectValid()

	// Gate enabled + field nil: required error
	st.Value(&Struct{}).Opts([]string{"MyGate"}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.Required(field.NewPath("stringPtrField"), ""),
		})

	// Gate disabled + field nil: valid (forbidden passes, optional short-circuits)
	st.Value(&Struct{}).ExpectValid()

	// Gate disabled + field set: forbidden error
	st.Value(&Struct{
		StringPtrField: ptr.To("Pod"),
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.Forbidden(field.NewPath("stringPtrField"), ""),
		})
}

func TestMultiGate(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Both gates enabled + field set: valid
	st.Value(&MultiGateStruct{
		MultiGatedField: ptr.To("val"),
	}).Opts([]string{"GateA", "GateB"}).ExpectValid()

	// Both gates enabled + field nil: required error
	st.Value(&MultiGateStruct{}).Opts([]string{"GateA", "GateB"}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.Required(field.NewPath("multiGatedField"), ""),
		})

	// Only GateA enabled + field set: forbidden (GateB not enabled)
	st.Value(&MultiGateStruct{
		MultiGatedField: ptr.To("val"),
	}).Opts([]string{"GateA"}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.Forbidden(field.NewPath("multiGatedField"), ""),
		})

	// No gates enabled + field nil: valid
	st.Value(&MultiGateStruct{}).ExpectValid()

	// No gates enabled + field set: forbidden
	st.Value(&MultiGateStruct{
		MultiGatedField: ptr.To("val"),
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.Forbidden(field.NewPath("multiGatedField"), ""),
		})
}

func TestSliceField(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Gate enabled + slice set: valid
	st.Value(&TypesStruct{
		SliceField: []string{"a"},
	}).Opts([]string{"SliceGate"}).ExpectValid()

	// Gate enabled + slice nil: valid (optional, no default)
	st.Value(&TypesStruct{}).Opts([]string{"SliceGate"}).ExpectValid()

	// Gate disabled + slice nil: valid
	st.Value(&TypesStruct{}).ExpectValid()

	// Gate disabled + slice set: forbidden
	st.Value(&TypesStruct{
		SliceField: []string{"a"},
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.Forbidden(field.NewPath("sliceField"), ""),
		})
}

func TestMapField(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Gate enabled + map set: valid
	st.Value(&TypesStruct{
		MapField: map[string]string{"k": "v"},
	}).Opts([]string{"MapGate"}).ExpectValid()

	// Gate enabled + map nil: valid (optional, no default)
	st.Value(&TypesStruct{}).Opts([]string{"MapGate"}).ExpectValid()

	// Gate disabled + map nil: valid
	st.Value(&TypesStruct{}).ExpectValid()

	// Gate disabled + map set: forbidden
	st.Value(&TypesStruct{
		MapField: map[string]string{"k": "v"},
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.Forbidden(field.NewPath("mapField"), ""),
		})
}

func TestValueField(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Gate enabled + value set: valid
	st.Value(&TypesStruct{
		ValueField: "hello",
	}).Opts([]string{"ValueGate"}).ExpectValid()

	// Gate enabled + value zero: valid (optional, no default)
	st.Value(&TypesStruct{}).Opts([]string{"ValueGate"}).ExpectValid()

	// Gate disabled + value zero: valid
	st.Value(&TypesStruct{}).ExpectValid()

	// Gate disabled + value set: forbidden
	st.Value(&TypesStruct{
		ValueField: "hello",
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.Forbidden(field.NewPath("valueField"), ""),
		})
}

func TestStructPtrField(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Gate enabled + struct set with valid inner: valid
	st.Value(&TypesStruct{
		StructPtrField: &InnerStruct{Name: "foo"},
	}).Opts([]string{"StructPtrGate"}).ExpectValid()

	// Gate enabled + struct set with invalid inner: inner validation error
	st.Value(&TypesStruct{
		StructPtrField: &InnerStruct{Name: ""},
	}).Opts([]string{"StructPtrGate"}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.TooShort(field.NewPath("structPtrField", "name"), "", 1),
		})

	// Gate enabled + struct nil: valid (optional)
	st.Value(&TypesStruct{}).Opts([]string{"StructPtrGate"}).ExpectValid()

	// Gate disabled + struct nil: valid
	st.Value(&TypesStruct{}).ExpectValid()

	// Gate disabled + struct set: forbidden
	st.Value(&TypesStruct{
		StructPtrField: &InnerStruct{Name: "foo"},
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField(),
		field.ErrorList{
			field.Forbidden(field.NewPath("structPtrField"), ""),
		})
}
