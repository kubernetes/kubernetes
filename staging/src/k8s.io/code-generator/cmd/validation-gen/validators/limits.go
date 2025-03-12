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

package validators

import (
	"fmt"
	"strconv"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/types"
)

const (
	minimumTagName          = "k8s:minimum"
	tightenedMinimumTagName = "k8s:tightenedMinimum"
)

func init() {
	RegisterTagValidator(minimumTagValidator{})
	RegisterTagValidator(tightenedMinimumTagValidator{})
}

type minimumTagValidator struct{}

func (minimumTagValidator) Init(_ Config) {}

func (minimumTagValidator) TagName() string {
	return minimumTagName
}

var minimumTagValidScopes = sets.New(
	ScopeAny,
)

func (minimumTagValidator) ValidScopes() sets.Set[Scope] {
	return minimumTagValidScopes
}

var (
	minimumValidator = types.Name{Package: libValidationPkg, Name: "Minimum"}
)

func (minimumTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	var result Validations

	if t := realType(context.Type); !types.IsInteger(t) {
		return result, fmt.Errorf("can only be used on integer types (%s)", rootTypeString(context.Type, t))
	}

	intVal, err := strconv.Atoi(payload)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %w", err)
	}
	result.AddFunction(Function(minimumTagName, DefaultFlags, minimumValidator, intVal))
	return result, nil
}

func (mtv minimumTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:         mtv.TagName(),
		Scopes:      mtv.ValidScopes().UnsortedList(),
		Description: "Indicates that a numeric field has a minimum value.",
		Payloads: []TagPayloadDoc{{
			Description: "<integer>",
			Docs:        "This field must be greater than or equal to x.",
		}},
	}
}

type tightenedMinimumTagValidator struct{}

func (tightenedMinimumTagValidator) Init(_ Config) {}

func (tightenedMinimumTagValidator) TagName() string {
	return tightenedMinimumTagName
}

var tightenedMinimumTagValidScopes = sets.New(
	ScopeAny,
)

func (tightenedMinimumTagValidator) ValidScopes() sets.Set[Scope] {
	return tightenedMinimumTagValidScopes
}

var (
	tightenedMinimumValidator = types.Name{Package: libValidationPkg, Name: "TightenedMinimum"}
)

func (tightenedMinimumTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	var result Validations

	if t := realType(context.Type); !types.IsInteger(t) {
		return result, fmt.Errorf("can only be used on integer types (%s)", rootTypeString(context.Type, t))
	}

	intVal, err := strconv.Atoi(payload)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %w", err)
	}
	result.AddFunction(Function(tightenedMinimumTagName, DefaultFlags, tightenedMinimumValidator, intVal))
	return result, nil
}

func (mtv tightenedMinimumTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:         mtv.TagName(),
		Scopes:      mtv.ValidScopes().UnsortedList(),
		Description: "Indicates that a numeric field has a minimum value that only applies if the old value is valid.",
		Payloads: []TagPayloadDoc{{
			Description: "<integer>",
			Docs:        "This field must be greater than or equal to x.",
		}},
	}
}
