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

package deepequalfunccrosspkg

import (
	"testing"

	"k8s.io/code-generator/cmd/validation-gen/output_tests/_codegenignore/other"
)

func TestDeepEqualFuncCrossPkg(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	val1 := 10
	st.Value(&Struct{
		NonComparableField: NonComparableStruct{Ptr: &val1},
	}).ExpectValidateFalseByPath(map[string][]string{
		"nonComparableField": {"field Struct.NonComparableField", "type NonComparableStruct"},
	})

	// Update with equal values ratchets through other.DeepEqual.
	other.DeepEqualCalls = 0
	oldObj := &Struct{NonComparableField: NonComparableStruct{Ptr: &val1}}
	newObj := &Struct{NonComparableField: NonComparableStruct{Ptr: &val1}}
	st.Value(newObj).OldValue(oldObj).ExpectValid()
	if other.DeepEqualCalls == 0 {
		t.Errorf("expected other.DeepEqual to be called during update ratcheting, got %d calls", other.DeepEqualCalls)
	}
}
