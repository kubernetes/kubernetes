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
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/codetags"
)

const (
	shadowTagName = "k8s:shadow"
)

func init() {
	RegisterTagValidator(&shadowTagValidator{})
}

type shadowTagValidator struct {
	validator Validator
}

func (stv *shadowTagValidator) Init(cfg Config) {
	stv.validator = cfg.Validator
}

func (shadowTagValidator) TagName() string {
	return shadowTagName
}

var shadowTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (shadowTagValidator) ValidScopes() sets.Set[Scope] {
	return shadowTagValidScopes
}

func (stv shadowTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	if tag.ValueType != codetags.ValueTypeTag || tag.ValueTag == nil {
		return Validations{}, fmt.Errorf("requires a validation tag as its value payload")
	}

	context.IsShadow = true
	validations, err := stv.validator.ExtractTagValidations(context, *tag.ValueTag)
	if err != nil {
		return Validations{}, err
	}

	result := Validations{}
	result.Variables = append(result.Variables, validations.Variables...)
	for _, fn := range validations.Functions {
		f := fn.WithShadow()
		result.AddFunction(f)
	}

	return result, nil
}

func (stv shadowTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            stv.TagName(),
		StabilityLevel: Alpha,
		Scopes:         stv.ValidScopes().UnsortedList(),
		Description:    "Marks the given payload validation as a shadow of the handwritten validation code.",
		Payloads: []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "The validation tag to evaluate as a shadow validation.",
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
	return doc
}
