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

// Package tags holds this project's validation tags. Importing it registers
// them with validation-gen's tag registry, so the generator's main package
// imports it for side effects.
package tags

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

// Tag names are relative to the generator's tag prefix: with prefix "xyz:",
// this tag is written +xyz:startsWith="value".
const startsWithTagName = "startsWith"

func init() {
	validators.RegisterTagValidator(startsWithTagValidator{})
}

type startsWithTagValidator struct{}

func (startsWithTagValidator) Init(_ validators.Config) {}

func (startsWithTagValidator) TagName() string {
	return startsWithTagName
}

var startsWithTagValidScopes = sets.New(validators.ScopeType, validators.ScopeField, validators.ScopeListVal, validators.ScopeMapKey, validators.ScopeMapVal)

func (startsWithTagValidator) ValidScopes() sets.Set[validators.Scope] {
	return startsWithTagValidScopes
}

var startsWithValidator = types.Name{Package: "k8s.io/code-generator/cmd/validation-gen/examples/custom-prefix/validate", Name: "StartsWith"}

func (v startsWithTagValidator) GetValidations(context validators.Context, tag codetags.Tag) (validators.Validations, error) {
	// This tag can apply to value and pointer fields, as well as typedefs.
	if t := util.NonPointer(util.NativeType(context.Type)); t != types.String {
		return validators.Validations{}, fmt.Errorf("can only be used on string types (%s)", t)
	}
	fn := validators.Function(startsWithTagName, validators.DefaultFlags, startsWithValidator, tag.Value).
		WithEmits(validators.Emission{Type: field.ErrorTypeInvalid, Origin: "startsWith"})
	return validators.Validations{Functions: []validators.FunctionGen{fn}}, nil
}

func (v startsWithTagValidator) Docs() validators.TagDoc {
	return validators.TagDoc{
		Tag:            v.TagName(),
		StabilityLevel: validators.TagStabilityLevelStable,
		Scopes:         sets.List(v.ValidScopes()),
		Description:    "Indicates that a string value must begin with the given prefix.",
		Payloads: []validators.TagPayloadDoc{{
			Description: "<string>",
			Docs:        "The required prefix.",
		}},
		PayloadsType:     codetags.ValueTypeString,
		PayloadsRequired: true,
	}
}
