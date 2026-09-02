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

package maps

import (
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Test(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&Struct{
		// All zero values
	}).ExpectValid()

	st.Value(&Struct{
		Max10MapField:                   map[string]string{strings.Repeat("a", 3): strings.Repeat("b", 3), strings.Repeat("c", 2): strings.Repeat("d", 2)},
	}).ExpectValid()

	testVal := &Struct{
		Max10MapField:                   map[string]string{strings.Repeat("a", 3): strings.Repeat("b", 3), strings.Repeat("c", 3): strings.Repeat("d", 2)},
	}
	st.Value(testVal).ExpectMatches(field.ErrorMatcher{}.ByType().ByField(), field.ErrorList{
		field.TooLong(field.NewPath("max10MapField"), "", 10),
	})

	// Test validation ratcheting
	st.Value(&Struct{
		Max10MapField:                   map[string]string{strings.Repeat("a", 3): strings.Repeat("b", 3), strings.Repeat("c", 3): strings.Repeat("d", 2)},
	}).OldValue(&Struct{
		Max10MapField:                   map[string]string{strings.Repeat("a", 3): strings.Repeat("b", 3), strings.Repeat("c", 3): strings.Repeat("d", 2)},
	}).ExpectValid()
}
