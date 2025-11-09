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
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	subfieldTagName = "k8s:subfield"
)

func init() {
	RegisterTagValidator(&subfieldTagValidator{})
}

type subfieldTagValidator struct {
	validator Validator
}

func (stv *subfieldTagValidator) Init(cfg Config) {
	stv.validator = cfg.Validator
}

func (subfieldTagValidator) TagName() string {
	return subfieldTagName
}

var subfieldTagValidScopes = sets.New(ScopeType, ScopeField, ScopeListVal, ScopeMapKey, ScopeMapVal)

func (subfieldTagValidator) ValidScopes() sets.Set[Scope] {
	return subfieldTagValidScopes
}

var (
	validateSubfield = types.Name{Package: libValidationPkg, Name: "Subfield"}
)

func (stv subfieldTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	args := tag.Args
	// This tag can apply to value and pointer fields, as well as typedefs
	// (which should never be pointers). We need to check the concrete type.
	t := context.Type
	nt := util.NonPointer(util.NativeType(t))
	if nt.Kind != types.Struct {
		return Validations{}, fmt.Errorf("can only be used on struct types: %v", nt.Kind)
	}
	subname := args[0].Value
	submemb := util.GetMemberByJSON(nt, subname)
	if submemb == nil {
		return Validations{}, fmt.Errorf("no field for json name %q", subname)
	}
	result := Validations{}
	subContext := Context{
		Scope:      ScopeField,
		Type:       submemb.Type,
		Path:       context.Path.Child(subname),
		Member:     submemb,
		ParentPath: context.Path,
	}
	if validations, err := stv.validator.ExtractValidations(subContext, *tag.ValueTag); err != nil {
		return Validations{}, err
	} else {
		if len(validations.Variables) > 0 {
			return Validations{}, fmt.Errorf("variable generation is not supported")
		}

		result.Variables = append(result.Variables, validations.Variables...)

		nilableStructType := context.Type
		if !util.IsNilableType(nilableStructType) {
			nilableStructType = types.PointerTo(nilableStructType)
		}

		nilableFieldType := submemb.Type
		fieldExprPrefix := ""
		if !util.IsNilableType(nilableFieldType) {
			nilableFieldType = types.PointerTo(nilableFieldType)
			fieldExprPrefix = "&"
		}

		// getFn is the function that retrieves the subfield value from the
		// struct.
		getFn := FunctionLiteral{
			Parameters: []ParamResult{{"o", nilableStructType}},
			Results:    []ParamResult{{"", nilableFieldType}},
		}
		getFn.Body = fmt.Sprintf("return %so.%s", fieldExprPrefix, submemb.Name)

		// equivArg is the function that is used to compare the correlated
		// elements in the old and new lists, for ratcheting.
		var equivArg any

		// directComparable is used to determine whether we can use the direct
		// comparison operator "==" or need to use the semantic DeepEqual when
		// looking up and comparing correlated list elements for validation ratcheting.
		directComparable := util.IsDirectComparable(util.NonPointer(util.NativeType(submemb.Type)))
		if directComparable {
			// It must be a pointer, since other nilable types are not directly
			// comparable.
			equivArg = Identifier(validateDirectEqualPtr)
		} else {
			equivArg = Identifier(validateSemanticDeepEqual)
		}

		for _, vfn := range validations.Functions {
			f := Function(subfieldTagName, vfn.Flags, validateSubfield, subname, getFn, equivArg, WrapperFunction{vfn, submemb.Type})
			f.Cohort = subname
			result.AddFunction(f)
		}
	}
	return result, nil
}

func (stv subfieldTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            stv.TagName(),
		StabilityLevel: Beta,
		Scopes:         stv.ValidScopes().UnsortedList(),
		Description:    "Declares a validation for a subfield of a struct.",
		Args: []TagArgDoc{{
			Description: "<field-json-name>",
			Type:        codetags.ArgTypeString,
			Required:    true,
		}},
		Docs: "The named subfield must be a direct field of the struct, or of an embedded struct.",
		Payloads: []TagPayloadDoc{{
			Description: "<validation-tag>",
			Docs:        "The tag to evaluate for the subfield.",
		}},
		PayloadsType:     codetags.ValueTypeTag,
		PayloadsRequired: true,
	}
	return doc
}
