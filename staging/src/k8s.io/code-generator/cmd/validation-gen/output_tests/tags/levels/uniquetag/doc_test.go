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

package uniquetag

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestMisc(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&UniqueStruct{
		AlphaUniqueSet: []string{"a", "a"},
		BetaUniqueSet:  []string{"a", "a"},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		field.Duplicate(field.NewPath("alphaUniqueSet").Index(1), "a").MarkAlpha(),
		field.Duplicate(field.NewPath("betaUniqueSet").Index(1), "a").MarkBeta(),
	})
}
