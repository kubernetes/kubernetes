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
	"k8s.io/gengo/v2/codetags"
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
	immutableValidator = types.Name{Package: libValidationPkg, Name: "Immutable"}
)

func (immutableTagValidator) GetValidations(context Context, _ codetags.Tag) (Validations, error) {
	var result Validations

	// Use ShortCircuit flag so immutable runs in the same group as +k8s:optional.
	result.AddFunction(Function(immutableTagName, ShortCircuit, immutableValidator))
	return result, nil
}

func (itv immutableTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            itv.TagName(),
		StabilityLevel: Alpha,
		Scopes:         itv.ValidScopes().UnsortedList(),
		Description:    "Indicates that a field may not be updated.",
	}
}
