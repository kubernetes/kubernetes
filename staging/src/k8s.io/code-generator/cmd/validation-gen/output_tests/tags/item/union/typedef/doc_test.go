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

package uniontypedef

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		Tasks: TaskList{
			{Name: "succeeded", State: "Succeeded"},
			{Name: "other", State: "Other"},
		},
	}).ExpectValid()

	invalidBothSet := &Struct{
		Tasks: []Task{
			{Name: "succeeded", State: "Succeeded"},
			{Name: "failed", State: "Failed"},
		},
	}

	st.Value(invalidBothSet).ExpectMatches(
		field.ErrorMatcher{},
		field.ErrorList{
			field.Invalid(field.NewPath("tasks"), "{TaskList[{\"name\": \"failed\"}], TaskList[{\"name\": \"succeeded\"}]}",
				"must specify exactly one of: `TaskList[{\"name\": \"succeeded\"}]`, `TaskList[{\"name\": \"failed\"}]`"),
		},
	)

	invalidEmpty := &Struct{
		Tasks: TaskList{},
	}
	st.Value(invalidEmpty).ExpectMatches(
		field.ErrorMatcher{},
		field.ErrorList{
			field.Invalid(field.NewPath("tasks"), "",
				"must specify one of: `TaskList[{\"name\": \"succeeded\"}]`, `TaskList[{\"name\": \"failed\"}]`"),
		},
	)

	// Test ratcheting.
	st.Value(invalidEmpty).OldValue(invalidEmpty).ExpectValid()

	st.Value(invalidBothSet).OldValue(invalidBothSet).ExpectValid()
}
