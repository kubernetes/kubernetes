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
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/types"
)

const (
	// These tags return a fixed pass/fail state.
	validateTrueTagName  = "k8s:validateTrue"
	validateFalseTagName = "k8s:validateFalse"

	// This tag always returns an error from ExtractValidations.
	validateErrorTagName = "k8s:validateError"
)

func init() {
	RegisterTagValidator(fixedResultTagValidator{result: true})
	RegisterTagValidator(fixedResultTagValidator{result: false})
	RegisterTagValidator(fixedResultTagValidator{error: true})
}

type fixedResultTagValidator struct {
	result bool
	error  bool
}

func (fixedResultTagValidator) Init(_ Config) {}

func (frtv fixedResultTagValidator) TagName() string {
	if frtv.error {
		return validateErrorTagName
	} else if frtv.result {
		return validateTrueTagName
	}
	return validateFalseTagName
}

var fixedResultTagValidScopes = sets.New(ScopeAny)

func (fixedResultTagValidator) ValidScopes() sets.Set[Scope] {
	return fixedResultTagValidScopes
}

func (frtv fixedResultTagValidator) GetValidations(context Context, _ []string, payload string) (Validations, error) {
	var result Validations

	if frtv.error {
		return result, fmt.Errorf("forced error: %q", payload)
	}

	tag, err := frtv.parseTagPayload(payload)
	if err != nil {
		return result, fmt.Errorf("can't decode tag payload: %w", err)
	}
	result.AddFunction(GenericFunction(frtv.TagName(), tag.flags, fixedResultValidator, tag.typeArgs, frtv.result, tag.msg))

	return result, nil
}

var (
	fixedResultValidator = types.Name{Package: libValidationPkg, Name: "FixedResult"}
)

type fixedResultPayload struct {
	flags    FunctionFlags
	msg      string
	typeArgs []types.Name
}

func (fixedResultTagValidator) parseTagPayload(in string) (fixedResultPayload, error) {
	type payload struct {
		Flags   []string `json:"flags"`
		Msg     string   `json:"msg"`
		TypeArg string   `json:"typeArg,omitempty"`
	}
	// We expect either a string (maybe empty) or a JSON object.
	if len(in) == 0 {
		return fixedResultPayload{}, nil
	}
	var pl payload
	if err := json.Unmarshal([]byte(in), &pl); err != nil {
		s := ""
		if err := json.Unmarshal([]byte(in), &s); err != nil {
			return fixedResultPayload{}, fmt.Errorf("error parsing JSON value: %w (%q)", err, in)
		}
		return fixedResultPayload{msg: s}, nil
	}
	// The msg field is required in JSON mode.
	if pl.Msg == "" {
		return fixedResultPayload{}, fmt.Errorf("JSON msg is required")
	}
	var flags FunctionFlags
	for _, fl := range pl.Flags {
		switch fl {
		case "ShortCircuit":
			flags |= ShortCircuit
		case "NonError":
			flags |= NonError
		default:
			return fixedResultPayload{}, fmt.Errorf("unknown flag: %q", fl)
		}
	}
	var typeArgs []types.Name
	if tn := pl.TypeArg; len(tn) > 0 {
		if !strings.HasPrefix(tn, "*") {
			tn = "*" + tn // We always need the pointer type.
		}
		typeArgs = []types.Name{{Package: "", Name: tn}}
	}

	return fixedResultPayload{flags, pl.Msg, typeArgs}, nil
}

func (frtv fixedResultTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:    frtv.TagName(),
		Scopes: frtv.ValidScopes().UnsortedList(),
	}
	if frtv.error {
		doc.Description = "Always fails code generation (useful for testing)."
		doc.Payloads = []TagPayloadDoc{{
			Description: "<string>",
			Docs:        "This string will be included in the error message.",
		}}
	} else {
		// True and false have the same payloads.
		doc.Payloads = []TagPayloadDoc{{
			Description: "<none>",
		}, {
			Description: "<quoted-string>",
			Docs:        "The generated code will include this string.",
		}, {
			Description: "<json-object>",
			Docs:        "",
			Schema: []TagPayloadSchema{{
				Key:   "flags",
				Value: "<list-of-flag-string>",
				Docs:  `values: ShortCircuit, NonError`,
			}, {
				Key:   "msg",
				Value: "<string>",
				Docs:  "The generated code will include this string.",
			}, {
				Key:   "typeArg",
				Value: "<string>",
				Docs:  "The type arg in generated code (must be the value-type, not pointer).",
			}},
		}}
		if frtv.result {
			doc.Description = "Always passes validation (useful for testing)."
		} else {
			doc.Description = "Always fails validation (useful for testing)."
		}
	}
	return doc
}
