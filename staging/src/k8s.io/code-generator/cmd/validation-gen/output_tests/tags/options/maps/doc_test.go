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

package maps

import (
	"testing"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		MapField: map[string]string{
			"a": "1",
		},
		MapFieldDisabled: map[string]string{
			"b": "2",
		},
		MapFieldEachKey: map[string]string{
			"a": "1",
		},
		MapFieldEachKeyDisabled: map[string]string{
			"b": "2",
		},
		MapFieldEachVal: map[string]string{
			"a": "1",
		},
		MapFieldEachValDisabled: map[string]string{
			"b": "2",
		},
	}).ExpectValidateFalseByPath(map[string][]string{
		"mapFieldDisabled":           {"field Struct.MapFieldDisabled"},
		"mapFieldEachKeyDisabled":    {"field Struct.MapFieldEachKeyDisabled/key"},
		"mapFieldEachValDisabled[b]": {"field Struct.MapFieldEachValDisabled/val"},
	})

	st.Value(&Struct{
		MapField: map[string]string{
			"a": "1",
		},
		MapFieldDisabled: map[string]string{
			"b": "2",
		},
		MapFieldEachKey: map[string]string{
			"a": "1",
		},
		MapFieldEachKeyDisabled: map[string]string{
			"b": "2",
		},
		MapFieldEachVal: map[string]string{
			"a": "1",
		},
		MapFieldEachValDisabled: map[string]string{
			"b": "2",
		},
	}).Opts([]string{"FeatureX"}).ExpectValidateFalseByPath(map[string][]string{
		"mapField":           {"field Struct.MapField"},
		"mapFieldEachKey":    {"field Struct.MapFieldEachKey/key"},
		"mapFieldEachVal[a]": {"field Struct.MapFieldEachVal/val"},
	})
}
