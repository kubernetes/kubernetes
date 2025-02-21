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
	formatTagName    = "k8s:format"
	maxLengthTagName = "k8s:maxLength"
	maxItemsTagName  = "k8s:maxItems"
)

func init() {
	RegisterTagValidator(formatTagValidator{})
	RegisterTagValidator(maxLengthTagValidator{})
	RegisterTagValidator(maxItemsTagValidator{})
}

type formatTagValidator struct{}

func (formatTagValidator) Init(_ Config) {}

func (formatTagValidator) TagName() string {
	return formatTagName
}

var formatTagValidScopes = sets.New(ScopeAny)

func (formatTagValidator) ValidScopes() sets.Set[Scope] {
	return formatTagValidScopes
}

var (
	ipSloppyValidator = types.Name{Package: libValidationPkg, Name: "IPSloppy"}
	dnsLabelValidator = types.Name{Package: libValidationPkg, Name: "DNSLabel"}
)

func (formatTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	var result Validations
	if formatFunction, err := getFormatValidationFunction(payload); err != nil {
		return result, err
	} else if formatFunction == nil {
		return result, fmt.Errorf("internal error: no validation function found for format %q", payload)
	} else {
		result.AddFunction(formatFunction)
	}
	return result, nil
}

func getFormatValidationFunction(format string) (FunctionGen, error) {
	// The naming convention for these formats follows the JSON schema style:
	// all lower-case, dashes between words. See
	// https://json-schema.org/draft/2020-12/json-schema-validation#name-defined-formats
	// for more examples.
	if format == "ip-sloppy" {
		return Function(formatTagName, DefaultFlags, ipSloppyValidator), nil
	}
	if format == "dns-label" {
		return Function(formatTagName, DefaultFlags, dnsLabelValidator), nil
	}
	// TODO: Flesh out the list of validation functions

	return nil, fmt.Errorf("unsupported validation format %q", format)
}

func (ftv formatTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:         ftv.TagName(),
		Scopes:      ftv.ValidScopes().UnsortedList(),
		Description: "Indicates that a string field has a particular format.",
		Payloads: []TagPayloadDoc{{
			Description: "ip-sloppy",
			Docs:        "This field holds an IPv4 or IPv6 address value. IPv4 octets may have leading zeros.",
		}, {
			Description: "dns-label",
			Docs:        "This field holds a DNS label value.",
		}},
	}
}

type maxLengthTagValidator struct{}

func (maxLengthTagValidator) Init(_ Config) {}

func (maxLengthTagValidator) TagName() string {
	return maxLengthTagName
}

var maxLengthTagValidScopes = sets.New(ScopeAny)

func (maxLengthTagValidator) ValidScopes() sets.Set[Scope] {
	return maxLengthTagValidScopes
}

var (
	maxLengthValidator = types.Name{Package: libValidationPkg, Name: "MaxLength"}
)

func (maxLengthTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	var result Validations

	t := context.Type
	if t.Kind == types.Alias {
		t = t.Underlying
	}
	if t != types.String {
		return result, fmt.Errorf("can only be used on string types")
	}

	intVal, err := strconv.Atoi(payload)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %v", err)
	}
	if intVal < 0 {
		return result, fmt.Errorf("must be greater than or equal to zero")
	}
	result.AddFunction(Function(maxLengthTagName, DefaultFlags, maxLengthValidator, intVal))
	return result, nil
}

func (mltv maxLengthTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:         mltv.TagName(),
		Scopes:      mltv.ValidScopes().UnsortedList(),
		Description: "Indicates that a string field has a limit on its length.",
		Payloads: []TagPayloadDoc{{
			Description: "<non-negative integer>",
			Docs:        "This field must be no more than X characters long.",
		}},
	}
}

type maxItemsTagValidator struct{}

func (maxItemsTagValidator) Init(_ Config) {}

func (maxItemsTagValidator) TagName() string {
	return maxItemsTagName
}

var maxItemsTagValidScopes = sets.New(
	ScopeType,
	ScopeField,
	ScopeListVal,
	ScopeMapVal,
)

func (maxItemsTagValidator) ValidScopes() sets.Set[Scope] {
	return maxItemsTagValidScopes
}

var (
	maxItemsValidator = types.Name{Package: libValidationPkg, Name: "MaxItems"}
)

func (maxItemsTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	var result Validations

	t := context.Type
	if t.Kind == types.Alias {
		t = t.Underlying
	}
	if t.Kind != types.Slice && t.Kind != types.Array {
		return result, fmt.Errorf("can only be used on list types")
	}

	intVal, err := strconv.Atoi(payload)
	if err != nil {
		return result, fmt.Errorf("failed to parse tag payload as int: %v", err)
	}
	if intVal < 0 {
		return result, fmt.Errorf("must be greater than or equal to zero")
	}
	result.AddFunction(Function(maxItemsTagName, ShortCircuit, maxItemsValidator, intVal))
	return result, nil
}

func (mitv maxItemsTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:         mitv.TagName(),
		Scopes:      mitv.ValidScopes().UnsortedList(),
		Description: "Indicates that a list field has a limit on its size.",
		Payloads: []TagPayloadDoc{
			{
				Description: "<non-negative integer>",
				Docs:        "This field must be no more than X items long.",
			},
		},
	}
}
