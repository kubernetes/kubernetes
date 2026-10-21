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

// +k8s:validation-gen=TypesWithField=TypeMeta
// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
// +k8s:validation-gen-deep-equal-func=CustomDeepEqual

// This is a test package.
// +k8s:validation-gen-nolint
package deepequalfunc

import (
	"reflect"

	"k8s.io/code-generator/cmd/validation-gen/testscheme"
)

var localSchemeBuilder = testscheme.New()

var CustomDeepEqualCalls int

// CustomDeepEqual deliberately disagrees with semantic deep-equal: for
// NonComparableStruct it compares only whether Ptr is set.  Tests use that
// difference to prove generated code honors this function's verdict.
func CustomDeepEqual[T any](a, b T) bool {
	CustomDeepEqualCalls++
	if x, ok := any(a).(*NonComparableStruct); ok {
		y := any(b).(*NonComparableStruct)
		if x == nil || y == nil {
			return x == y
		}
		return (x.Ptr == nil) == (y.Ptr == nil)
	}
	return reflect.DeepEqual(a, b)
}

type Struct struct {
	TypeMeta int

	// +k8s:validateFalse="field Struct.NonComparableField"
	NonComparableField NonComparableStruct `json:"nonComparableField"`

	// +k8s:eachVal=+k8s:validateFalse="field Struct.MapField[*]"
	MapField map[string]NonComparableStruct `json:"mapField"`

	// +k8s:listType=set
	// +k8s:eachVal=+k8s:validateFalse="field Struct.SetField[*]"
	SetField []NonComparableStruct `json:"setField"`
}

// +k8s:validateFalse="type NonComparableStruct"
type NonComparableStruct struct {
	// Ptr makes it not direct-comparable
	Ptr *int `json:"ptr"`
}
