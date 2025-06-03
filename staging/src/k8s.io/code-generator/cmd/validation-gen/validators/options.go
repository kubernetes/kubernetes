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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/codetags"
)

const (
	ifOptionEnabledTag  = "k8s:ifOptionEnabled"
	ifOptionDisabledTag = "k8s:ifOptionDisabled"
)

func init() {
	RegisterTagValidator(&ifOptionTagValidator{true, nil})
	RegisterTagValidator(&ifOptionTagValidator{false, nil})
}

type ifOptionTagValidator struct {
	enabled   bool
	validator Validator
}

func (iotv *ifOptionTagValidator) Init(cfg Config) {
	iotv.validator = cfg.Validator
}

func (iotv ifOptionTagValidator) TagName() string {
	if iotv.enabled {
		return ifOptionEnabledTag
	}
	return ifOptionDisabledTag
}

var ifOptionTagValidScopes = sets.New(ScopeAny)

func (ifOptionTagValidator) ValidScopes() sets.Set[Scope] {
	return ifOptionTagValidScopes
}

func (iotv ifOptionTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	if len(tag.Args) != 1 || len(tag.Args[0].Name) != 0 || tag.Args[0].Type != codetags.ArgTypeString {
		return Validations{}, fmt.Errorf("requires exactly 1 positional string argument")
	}
	optionName := tag.Args[0].Value
	if tag.ValueTag == nil {
		return Validations{}, fmt.Errorf("missing validation tag")
	}

	result := Validations{}
	if validations, err := iotv.validator.ExtractValidations(context, *tag.ValueTag); err != nil {
		return Validations{}, err
	} else {
		for _, fn := range validations.Functions {
			if iotv.enabled {
				result.Functions = append(result.Functions, fn.WithConditions(Conditions{OptionEnabled: optionName}))
			} else {
				result.Functions = append(result.Functions, fn.WithConditions(Conditions{OptionDisabled: optionName}))
			}
			result.Variables = append(result.Variables, validations.Variables...)
		}
		return result, nil
	}
}

func (iotv ifOptionTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag: iotv.TagName(),
		Args: []TagArgDoc{{
			Description: "<option>",
		}},
		Scopes: iotv.ValidScopes().UnsortedList(),
	}

	if iotv.enabled {
		doc.Description = "Declares a validation that only applies when an option is enabled."
		doc.Payloads = []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "This validation tag will be evaluated only if the validation option is enabled.",
		}}
	} else {
		doc.Description = "Declares a validation that only applies when an option is disabled."
		doc.Payloads = []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "This validation tag will be evaluated only if the validation option is disabled.",
		}}
	}
	return doc
}
