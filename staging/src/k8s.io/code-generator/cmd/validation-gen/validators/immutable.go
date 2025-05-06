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

package validators

import (
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/types"
)

const (
	immutableTagName = "k8s:immutable"
)

func init() {
	RegisterTagValidator(immutableTagValidator{})
}

type immutableTagValidator struct{}

func (immutableTagValidator) Init(_ Config) {}

func (immutableTagValidator) TagName() string {
	return immutableTagName
}

var immutableTagValidScopes = sets.New(ScopeField, ScopeType, ScopeMapVal, ScopeListVal)

func (immutableTagValidator) ValidScopes() sets.Set[Scope] {
	return immutableTagValidScopes
}

var (
	immutableCompareValidator = types.Name{Package: libValidationPkg, Name: "ImmutableByCompare"}
	immutableReflectValidator = types.Name{Package: libValidationPkg, Name: "ImmutableByReflect"}
)

func (immutableTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	var result Validations

	if nonPointer(nativeType(context.Type)).Kind == types.Builtin {
		// This is a minor optimization to just compare primitive values when
		// possible. Slices and maps are not comparable, and structs might hold
		// pointer fields, which are directly comparable but not what we need.
		//
		// Note: This compares the pointee, not the pointer itself.
		result.AddFunction(Function(immutableTagName, DefaultFlags, immutableCompareValidator))
	} else {
		result.AddFunction(Function(immutableTagName, DefaultFlags, immutableReflectValidator))
	}

	return result, nil
}

func (itv immutableTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:         itv.TagName(),
		Scopes:      itv.ValidScopes().UnsortedList(),
		Description: "Indicates that a field may not be updated.",
	}
}
