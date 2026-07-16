/*
Copyright 2024 The Kubernetes Authors.

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

package simple

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{ /* zero values */ }).ExpectValid()

	st.Value(&Struct{D: DM1, M1: &M1{}}).ExpectValid()
	st.Value(&Struct{D: DM2, M2: &M2{}}).ExpectValid()
	st.Value(&Struct{D: DM3, M3: []string{"a"}}).ExpectValid()
	st.Value(&Struct{D: DM4, M4: map[string]string{"k": "v"}}).ExpectValid()

	st.Value(&Struct{D: DM2, M1: &M1{}, M2: &M2{}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m1"), nil, "may only be specified when"),
	}.WithOrigin("union"))

	st.Value(&Struct{D: DM1}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m1"), nil, "must be specified when"),
	}.WithOrigin("union"))

	// Slice member: discriminator matches but slice is nil
	st.Value(&Struct{D: DM3}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m3"), nil, "must be specified when"),
	}.WithOrigin("union"))
	// Slice member: discriminator matches but slice is empty (len==0, not nil)
	st.Value(&Struct{D: DM3, M3: []string{}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m3"), nil, "must be specified when"),
	}.WithOrigin("union"))

	// Slice member: discriminator doesn't match but slice is set
	st.Value(&Struct{D: DM1, M1: &M1{}, M3: []string{"a"}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m3"), nil, "may only be specified when"),
	}.WithOrigin("union"))

	// Map member: discriminator matches but map is nil
	st.Value(&Struct{D: DM4}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m4"), nil, "must be specified when"),
	}.WithOrigin("union"))
	// Map member: discriminator matches but map is empty (len==0, not nil)
	st.Value(&Struct{D: DM4, M4: map[string]string{}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m4"), nil, "must be specified when"),
	}.WithOrigin("union"))

	// Map member: discriminator doesn't match but map is set
	st.Value(&Struct{D: DM1, M1: &M1{}, M4: map[string]string{"k": "v"}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m4"), nil, "may only be specified when"),
	}.WithOrigin("union"))

	// Test validation ratcheting
	st.Value(&Struct{D: DM2, M1: &M1{}, M2: &M2{}}).OldValue(&Struct{D: DM2, M1: &M1{}, M2: &M2{}}).ExpectValid()
	st.Value(&Struct{D: DM1}).OldValue(&Struct{D: DM1}).ExpectValid()

	// Slice member ratcheting: unchanged membership
	st.Value(&Struct{D: DM3, M3: []string{"a"}}).OldValue(&Struct{D: DM3, M3: []string{"b"}}).ExpectValid()
	// Slice member ratcheting: changed from set to unset
	st.Value(&Struct{D: DM3}).OldValue(&Struct{D: DM3, M3: []string{"a"}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m3"), nil, "must be specified when"),
	}.WithOrigin("union"))

	// Map member ratcheting: unchanged membership
	st.Value(&Struct{D: DM4, M4: map[string]string{"k": "v1"}}).OldValue(&Struct{D: DM4, M4: map[string]string{"k": "v2"}}).ExpectValid()
	// Map member ratcheting: changed from set to unset
	st.Value(&Struct{D: DM4}).OldValue(&Struct{D: DM4, M4: map[string]string{"k": "v"}}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m4"), nil, "must be specified when"),
	}.WithOrigin("union"))

	// Test update with nil old value (simulates newly-set pointer field during update).
	// Discriminated union validation should still detect mismatches even though oldObj is nil.
	st.Value(&Struct{D: DM1}).OldValue(nil).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring().ByOrigin(), field.ErrorList{
		field.Invalid(field.NewPath("m1"), nil, "must be specified when"),
	}.WithOrigin("union"))
}
