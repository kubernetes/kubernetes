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

package equalto

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestStringEqual(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Both fields equal → valid.
	st.Value(&StringEqual{Name: "same", TargetName: "same"}).ExpectValid()

	// Both zero → valid (zero == zero).
	st.Value(&StringEqual{}).ExpectValid()

	// Fields differ → error on the tagged field.
	st.Value(&StringEqual{Name: "aaa", TargetName: "bbb"}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Invalid(field.NewPath("name"), "", "").WithOrigin("equalTo"),
		},
	)

	// Ratchet: neither changed → skip.
	st.Value(&StringEqual{Name: "aaa", TargetName: "bbb"}).
		OldValue(&StringEqual{Name: "aaa", TargetName: "bbb"}).
		ExpectValid()

	// Update: value changed, now differs → fire.
	st.Value(&StringEqual{Name: "new", TargetName: "bbb"}).
		OldValue(&StringEqual{Name: "old", TargetName: "bbb"}).
		ExpectMatches(
			field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
			field.ErrorList{
				field.Invalid(field.NewPath("name"), "", "").WithOrigin("equalTo"),
			},
		)
}

func TestIntEqual(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&IntEqual{HostPort: 80, Port: 80}).ExpectValid()

	st.Value(&IntEqual{HostPort: 80, Port: 443}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Invalid(field.NewPath("hostPort"), "", "").WithOrigin("equalTo"),
		},
	)
}

func TestPtrEqual(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	a := "same"
	b := "same"
	st.Value(&PtrEqual{Field: &a, Other: &b}).ExpectValid()

	c := "aaa"
	d := "bbb"
	st.Value(&PtrEqual{Field: &c, Other: &d}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Invalid(field.NewPath("field"), "", "").WithOrigin("equalTo"),
		},
	)

	// Both nil (zero values for pointer) → equal.
	st.Value(&PtrEqual{}).ExpectValid()
}

func TestBidirectionalEqual(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	// Both equal → valid.
	st.Value(&BidirectionalEqual{FieldA: "same", FieldB: "same"}).ExpectValid()

	// Differ → errors on both tagged fields.
	st.Value(&BidirectionalEqual{FieldA: "aaa", FieldB: "bbb"}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{
			field.Invalid(field.NewPath("fieldA"), "", "").WithOrigin("equalTo"),
			field.Invalid(field.NewPath("fieldB"), "", "").WithOrigin("equalTo"),
		},
	)
}
