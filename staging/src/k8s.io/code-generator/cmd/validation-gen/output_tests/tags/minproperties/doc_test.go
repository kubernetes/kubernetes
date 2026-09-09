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

package minproperties

import (
	"fmt"
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
		Min0Field:         make(map[string]string, 0),
		Min10Field:        make(map[string]string, 0),
		Min0TypedefField:  make(map[string]StringType, 0),
		Min10TypedefField: make(map[string]StringType, 0),
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooFew(field.NewPath("min10Field"), 0, 10),
		field.TooFew(field.NewPath("min10TypedefField"), 0, 10),
	})

	min10Field1 := make(map[string]string)
	min10TypedefField1 := make(map[string]StringType)
	for i := range 1 {
		min10Field1[fmt.Sprintf("k%d", i)] = "v"
		min10TypedefField1[fmt.Sprintf("k%d", i)] = "v"
	}

	st.Value(&Struct{
		Min10Field:        min10Field1,
		Min10TypedefField: min10TypedefField1,
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooFew(field.NewPath("min10Field"), 1, 10),
		field.TooFew(field.NewPath("min10TypedefField"), 1, 10),
	})

	min10Field9 := make(map[string]string)
	min10TypedefField9 := make(map[string]StringType)
	for i := range 9 {
		min10Field9[fmt.Sprintf("k%d", i)] = "v"
		min10TypedefField9[fmt.Sprintf("k%d", i)] = "v"
	}

	st.Value(&Struct{
		Min10Field:        min10Field9,
		Min10TypedefField: min10TypedefField9,
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooFew(field.NewPath("min10Field"), 9, 10),
		field.TooFew(field.NewPath("min10TypedefField"), 9, 10),
	})

	min10Field10 := make(map[string]string)
	min10TypedefField10 := make(map[string]StringType)
	for i := range 10 {
		min10Field10[fmt.Sprintf("k%d", i)] = "v"
		min10TypedefField10[fmt.Sprintf("k%d", i)] = "v"
	}

	st.Value(&Struct{
		Min10Field:        min10Field10,
		Min10TypedefField: min10TypedefField10,
	}).ExpectValid()

	min0Field1 := make(map[string]string)
	min10Field11 := make(map[string]string)
	min0TypedefField1 := make(map[string]StringType)
	min10TypedefField11 := make(map[string]StringType)

	for i := range 1 {
		min0Field1[fmt.Sprintf("k%d", i)] = "v"
		min0TypedefField1[fmt.Sprintf("k%d", i)] = "v"
	}
	for i := range 11 {
		min10Field11[fmt.Sprintf("k%d", i)] = "v"
		min10TypedefField11[fmt.Sprintf("k%d", i)] = "v"
	}

	testVal := &Struct{
		Min0Field:         min0Field1,
		Min10Field:        min10Field11,
		Min0TypedefField:  min0TypedefField1,
		Min10TypedefField: min10TypedefField11,
	}
	st.Value(testVal).ExpectValid()

	// Test validation ratcheting
	st.Value(&Struct{
		Min0Field:         min0Field1,
		Min10Field:        min10Field11,
		Min0TypedefField:  min0TypedefField1,
		Min10TypedefField: min10TypedefField11,
	}).OldValue(&Struct{
		Min0Field:         min0Field1,
		Min10Field:        min10Field1,
		Min0TypedefField:  min0TypedefField1,
		Min10TypedefField: min10TypedefField1,
	}).ExpectValid()

}
