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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/gengo/v2/codetags"
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

func (frtv fixedResultTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	var result Validations

	if frtv.error {
		return result, fmt.Errorf("forced error: %q", tag.Value)
	}

	args, err := frtv.toFixedResultArgs(tag)
	if err != nil {
		return result, fmt.Errorf("can't decode tag payload: %w", err)
	}
	result.AddFunction(Function(frtv.TagName(), args.flags, fixedResultValidator, frtv.result, args.msg).WithTypeArgs(args.typeArgs...))

	return result, nil
}

var (
	fixedResultValidator = types.Name{Package: libValidationPkg, Name: "FixedResult"}
)

type fixedResultArgs struct {
	flags    FunctionFlags
	msg      string
	typeArgs []types.Name
}

func (fixedResultTagValidator) toFixedResultArgs(in codetags.Tag) (fixedResultArgs, error) {
	result := fixedResultArgs{}
	for _, a := range in.Args {
		switch a.Name {
		case "flags":
			for _, fl := range strings.Split(a.Value, ",") {
				fl = strings.TrimSpace(fl)
				switch fl {
				case "ShortCircuit":
					result.flags |= ShortCircuit
				case "NonError":
					result.flags |= NonError
				default:
					return fixedResultArgs{}, fmt.Errorf("unknown flag: %q", fl)
				}
			}
		case "typeArg":
			if tn := a.Value; len(tn) > 0 {
				if !strings.HasPrefix(tn, "*") {
					tn = "*" + tn // We always need the pointer type.
				}
				result.typeArgs = []types.Name{{Package: "", Name: tn}}
			}
		}
	}
	if in.ValueType == codetags.ValueTypeString {
		result.msg = in.Value
	}
	return result, nil
}

func (frtv fixedResultTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:    frtv.TagName(),
		Scopes: frtv.ValidScopes().UnsortedList(),
	}
	doc.PayloadsType = codetags.ValueTypeString
	if frtv.error {
		doc.Description = "Always fails code generation (useful for testing)."
		doc.Payloads = []TagPayloadDoc{{
			Description: "<string>",
			Docs:        "This string will be included in the error message.",
		}}
	} else {
		// True and false have the same args and payload.
		doc.Payloads = []TagPayloadDoc{{
			Description: "<none>",
		}, {
			Description: "<string>",
			Docs:        "The generated code will include this string.",
		}}
		doc.Args = []TagArgDoc{{
			Name:        "flags",
			Description: "<comma-separated-list-of-flag-string>",
			Docs:        `values: ShortCircuit, NonError`,
			Type:        codetags.ArgTypeString,
		}, {
			Name:        "typeArg",
			Description: "<string>",
			Docs:        "The type arg in generated code (must be the value-type, not pointer).",
			Type:        codetags.ArgTypeString,
		}}
		if frtv.result {
			doc.Description = "Always passes validation (useful for testing)."
		} else {
			doc.Description = "Always fails validation (useful for testing)."
		}
	}
	return doc
}
