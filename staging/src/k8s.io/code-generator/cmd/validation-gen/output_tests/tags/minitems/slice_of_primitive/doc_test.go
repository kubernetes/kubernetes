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

package sliceofprimitive

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooFew(field.NewPath("min10Field"), 0, 10),
		field.TooFew(field.NewPath("min10TypedefField"), 0, 10),
	})

	st.Value(&Struct{
		Min0Field:         make([]int, 0),
		Min10Field:        make([]int, 0),
		Min0TypedefField:  make([]IntType, 0),
		Min10TypedefField: make([]IntType, 0),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooFew(field.NewPath("min10Field"), 0, 10),
		field.TooFew(field.NewPath("min10TypedefField"), 0, 10),
	})

	st.Value(&Struct{
		Min10Field:        make([]int, 1),
		Min10TypedefField: make([]IntType, 1),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooFew(field.NewPath("min10Field"), 1, 10),
		field.TooFew(field.NewPath("min10TypedefField"), 1, 10),
	})

	st.Value(&Struct{
		Min10Field:        make([]int, 9),
		Min10TypedefField: make([]IntType, 9),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooFew(field.NewPath("min10Field"), 9, 10),
		field.TooFew(field.NewPath("min10TypedefField"), 9, 10),
	})

	st.Value(&Struct{
		Min10Field:        make([]int, 10),
		Min10TypedefField: make([]IntType, 10),
	}).ExpectValid()

	testVal := &Struct{
		Min0Field:         make([]int, 1),
		Min10Field:        make([]int, 11),
		Min0TypedefField:  make([]IntType, 1),
		Min10TypedefField: make([]IntType, 11),
	}
	st.Value(testVal).ExpectValid()

	// Test validation ratcheting
	st.Value(&Struct{
		Min0Field:         make([]int, 1),
		Min10Field:        make([]int, 1),
		Min0TypedefField:  make([]IntType, 1),
		Min10TypedefField: make([]IntType, 1),
	}).OldValue(&Struct{
		Min0Field:         make([]int, 1),
		Min10Field:        make([]int, 1),
		Min0TypedefField:  make([]IntType, 1),
		Min10TypedefField: make([]IntType, 1),
	}).ExpectValid()

}
