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

package lists

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		ListMap: []ListItem{
			{Name: "a", Value: "1"},
			{Name: "a", Value: "2"},
		},
		ListMapDisabled: []ListItem{
			{Name: "b", Value: "1"},
			{Name: "b", Value: "2"},
		},
	}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{field.Duplicate(field.NewPath("listMapDisabled").Index(1), ListItem{Name: "b", Value: "2"}).WithOrigin("")},
	)

	st.Value(&Struct{
		ListMap: []ListItem{
			{Name: "a", Value: "1"},
			{Name: "a", Value: "2"},
		},
		ListMapDisabled: []ListItem{
			{Name: "b", Value: "1"},
			{Name: "b", Value: "2"},
		},
	}).Opts([]string{"FeatureX"}).ExpectMatches(
		field.ErrorMatcher{}.ByType().ByField().ByOrigin(),
		field.ErrorList{field.Duplicate(field.NewPath("listMap").Index(1), ListItem{Name: "a", Value: "2"}).WithOrigin("")},
	)

	st.Value(&Struct{
		ListEachVal: []ListItem{
			{Name: "c", Value: "3"},
		},
		ListEachValDisabled: []ListItem{
			{Name: "d", Value: "4"},
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"listEachValDisabled[0]": {"field Struct.ListEachValDisabled/val"},
	})

	st.Value(&Struct{
		ListEachVal: []ListItem{
			{Name: "c", Value: "3"},
		},
		ListEachValDisabled: []ListItem{
			{Name: "d", Value: "4"},
		},
	}).Opts([]string{"FeatureX"}).ExpectValidateFalseByPath(map[string][]string{
		"listEachVal[0]": {"field Struct.ListEachVal/val"},
	})
}
