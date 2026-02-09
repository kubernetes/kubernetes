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

package listkeys

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestListKeys(t *testing.T) {
	st := localSchemeBuilder.Test(t)

	st.Value(&ListKeyStruct{
		// Case 1: Alpha listType, Standard Key
		AlphaListTypeStandardKey: []MapItem{
			{Key: "a", Value: 1},
			{Key: "a", Value: 2},
		},
		// Case 2: Standard listType, Alpha Key
		StandardListTypeAlphaKey: []MapItem{
			{Key: "b", Value: 1},
			{Key: "b", Value: 2},
		},
		// Case 3: Alpha listType, Alpha Key
		AlphaListTypeAlphaKey: []MapItem{
			{Key: "c", Value: 1},
			{Key: "c", Value: 2},
		},
		// Case 4: Standard listType, Alpha Key1, Standard Key2
		StandardListTypeMixedKeys1: []MultiKeyItem{
			{Key1: "d", Key2: 1, Value: 1},
			{Key1: "d", Key2: 1, Value: 2},
		},
		// Case 5: Standard listType, Standard Key1, Alpha Key2
		StandardListTypeMixedKeys2: []MultiKeyItem{
			{Key1: "e", Key2: 1, Value: 1},
			{Key1: "e", Key2: 1, Value: 2},
		},
		// Case 6: Alpha listType, Alpha Key1, Standard Key2
		AlphaListTypeMixedKeys1: []MultiKeyItem{
			{Key1: "f", Key2: 1, Value: 1},
			{Key1: "f", Key2: 1, Value: 2},
		},
		// Case 7: Alpha listType, Standard Key1, Alpha Key2
		AlphaListTypeMixedKeys2: []MultiKeyItem{
			{Key1: "g", Key2: 1, Value: 1},
			{Key1: "g", Key2: 1, Value: 2},
		},

		// Case 11: Standard listType, Beta Key1, Standard Key2
		StandardListTypeMixedKeysBeta1: []MultiKeyItem{
			{Key1: "k", Key2: 1, Value: 1},
			{Key1: "k", Key2: 1, Value: 2},
		},
		// Case 12: Standard listType, Standard Key1, Beta Key2
		StandardListTypeMixedKeysBeta2: []MultiKeyItem{
			{Key1: "l", Key2: 1, Value: 1},
			{Key1: "l", Key2: 1, Value: 2},
		},
		// Case 13: Beta listType, Beta Key1, Standard Key2
		BetaListTypeMixedKeys1: []MultiKeyItem{
			{Key1: "m", Key2: 1, Value: 1},
			{Key1: "m", Key2: 1, Value: 2},
		},
		// Case 14: Beta listType, Standard Key1, Beta Key2
		BetaListTypeMixedKeys2: []MultiKeyItem{
			{Key1: "n", Key2: 1, Value: 1},
			{Key1: "n", Key2: 1, Value: 2},
		},

		// Case 8: Beta listType, Standard Key
		BetaListTypeStandardKey: []MapItem{
			{Key: "h", Value: 1},
			{Key: "h", Value: 2},
		},
		// Case 9: Standard listType, Beta Key
		StandardListTypeBetaKey: []MapItem{
			{Key: "i", Value: 1},
			{Key: "i", Value: 2},
		},
		// Case 10: Beta listType, Beta Key
		BetaListTypeBetaKey: []MapItem{
			{Key: "j", Value: 1},
			{Key: "j", Value: 2},
		},
	}).ExpectMatches(field.ErrorMatcher{}.ByType().ByField().ByOrigin().ByValidationStabilityLevel(), field.ErrorList{
		// Case 1: Alpha listType, Standard Key -> Alpha Error
		field.Duplicate(field.NewPath("alphaListTypeStandardKey").Index(1), MapItem{Key: "a", Value: 2}).MarkAlpha(),

		// Case 2: Standard listType, Alpha Key -> Normal Error
		// The alpha key is treated as a functional key for the list validation.
		field.Duplicate(field.NewPath("standardListTypeAlphaKey").Index(1), MapItem{Key: "b", Value: 2}),

		// Case 3: Alpha listType, Alpha Key -> Alpha Error
		field.Duplicate(field.NewPath("alphaListTypeAlphaKey").Index(1), MapItem{Key: "c", Value: 2}).MarkAlpha(),

		// Case 4: Standard listType, Alpha Key1, Standard Key2 -> Normal Error
		// Both keys participate in the uniqueness check.
		field.Duplicate(field.NewPath("standardListTypeMixedKeys1").Index(1), MultiKeyItem{Key1: "d", Key2: 1, Value: 2}),

		// Case 5: Standard listType, Standard Key1, Alpha Key2 -> Normal Error
		field.Duplicate(field.NewPath("standardListTypeMixedKeys2").Index(1), MultiKeyItem{Key1: "e", Key2: 1, Value: 2}),

		// Case 6: Alpha listType, Alpha Key1, Standard Key2 -> Alpha Error
		field.Duplicate(field.NewPath("alphaListTypeMixedKeys1").Index(1), MultiKeyItem{Key1: "f", Key2: 1, Value: 2}).MarkAlpha(),

		// Case 7: Alpha listType, Standard Key1, Alpha Key2 -> Alpha Error
		field.Duplicate(field.NewPath("alphaListTypeMixedKeys2").Index(1), MultiKeyItem{Key1: "g", Key2: 1, Value: 2}).MarkAlpha(),

		// Case 11: Standard listType, Beta Key1, Standard Key2 -> Normal Error
		field.Duplicate(field.NewPath("standardListTypeMixedKeysBeta1").Index(1), MultiKeyItem{Key1: "k", Key2: 1, Value: 2}),

		// Case 12: Standard listType, Standard Key1, Beta Key2 -> Normal Error
		field.Duplicate(field.NewPath("standardListTypeMixedKeysBeta2").Index(1), MultiKeyItem{Key1: "l", Key2: 1, Value: 2}),

		// Case 13: Beta listType, Beta Key1, Standard Key2 -> Beta Error
		field.Duplicate(field.NewPath("betaListTypeMixedKeys1").Index(1), MultiKeyItem{Key1: "m", Key2: 1, Value: 2}).MarkBeta(),

		// Case 14: Beta listType, Standard Key1, Beta Key2 -> Beta Error
		field.Duplicate(field.NewPath("betaListTypeMixedKeys2").Index(1), MultiKeyItem{Key1: "n", Key2: 1, Value: 2}).MarkBeta(),

		// Case 8: Beta listType, Standard Key -> Beta Error
		field.Duplicate(field.NewPath("betaListTypeStandardKey").Index(1), MapItem{Key: "h", Value: 2}).MarkBeta(),

		// Case 9: Standard listType, Beta Key -> Normal Error
		// The beta key is treated as a functional key for the list validation.
		field.Duplicate(field.NewPath("standardListTypeBetaKey").Index(1), MapItem{Key: "i", Value: 2}),

		// Case 10: Beta listType, Beta Key -> Beta Error
		field.Duplicate(field.NewPath("betaListTypeBetaKey").Index(1), MapItem{Key: "j", Value: 2}).MarkBeta(),
	})
}
