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
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	opaqueTypeTagName = "k8s:opaqueType"
)

// globalOpaqueTypes and globalOpaqueMembers record the set of types and struct
// members tagged with +k8s:opaqueType, populated as opaqueTypeTagValidator
// processes those tags. Other validators (currently subfield) consult these to
// decide whether to inherit short-circuit validations from the field. Lookups
// must therefore happen in a deferred callback so that all opaque tags have
// been registered first.
var (
	globalOpaqueTypes   = map[*types.Type]bool{}
	globalOpaqueMembers = map[*types.Member]bool{}
)

type opaqueTypeTagValidator struct{}

func init() {
	RegisterTagValidator(opaqueTypeTagValidator{})
}

func (opaqueTypeTagValidator) Init(Config) {}

func (opaqueTypeTagValidator) TagName() string {
	return opaqueTypeTagName
}

func (opaqueTypeTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)
}

func (opaqueTypeTagValidator) GetValidations(context Context, _ codetags.Tag) (Validations, error) {
	// Store information about opaque types and fields, so that other validators
	// don't have to extract it again.
	switch context.Scope {
	case ScopeType:
		globalOpaqueTypes[util.NonPointer(util.NativeType(context.Type))] = true
	case ScopeField:
		if context.Member != nil {
			globalOpaqueMembers[context.Member] = true
		}
	}
	return Validations{OpaqueType: true}, nil
}

// isFieldOpaque reports whether the field or its type was tagged
// +k8s:opaqueType. It reads from the package-level registries above, so
// callers must invoke it from a deferred callback (or any other context
// where opaque-tag processing has finished).
func isFieldOpaque(context Context) bool {
	if context.Member != nil && globalOpaqueMembers[context.Member] {
		return true
	}
	if context.Type != nil && globalOpaqueTypes[util.NonPointer(util.NativeType(context.Type))] {
		return true
	}
	return false
}

func (v opaqueTypeTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            opaqueTypeTagName,
		StabilityLevel: TagStabilityLevelStable,
		Scopes:         sets.List(v.ValidScopes()),
		Description: "Indicates that any validations declared on the referenced type will be ignored. " +
			"If a referenced type's package is not included in the generator's current " +
			"flags, this tag must be set, or code generation will fail (preventing silent " +
			"mistakes). If the validations should not be ignored, add the type's package " +
			"to the generator using the --readonly-pkg flag.",
	}
	return doc
}
