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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/parser/tags"
	"k8s.io/gengo/v2/types"
)

const (
	equalToTagName = "k8s:equalTo"
)

func init() {
	RegisterTagValidator(equalToTagValidator{})
}

type equalToTagValidator struct{}

func (equalToTagValidator) Init(_ Config) {}

func (equalToTagValidator) TagName() string {
	return equalToTagName
}

var equalToTagValidScopes = sets.New(ScopeField)

func (equalToTagValidator) ValidScopes() sets.Set[Scope] {
	return equalToTagValidScopes
}

var equalToValidator = types.Name{Package: libValidationPkg, Name: "EqualTo"}

func (v equalToTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	if context.Member == nil {
		return Validations{}, fmt.Errorf("must be used on a struct field")
	}

	parentType := util.NonPointer(util.NativeType(context.ParentType))
	if parentType.Kind != types.Struct {
		return Validations{}, fmt.Errorf("parent type must be a struct (got %s)", parentType.Kind)
	}

	if len(tag.Args) != 1 {
		return Validations{}, fmt.Errorf("expected exactly one argument naming the sibling field, got %d", len(tag.Args))
	}
	siblingJSONName := tag.Args[0].Value

	siblingMember := util.GetMemberByJSON(parentType, siblingJSONName)
	if siblingMember == nil {
		return Validations{}, fmt.Errorf("no sibling field with JSON name %q", siblingJSONName)
	}
	if siblingMember.Name == context.Member.Name {
		return Validations{}, fmt.Errorf("field cannot reference itself (both are %q)", siblingJSONName)
	}

	fieldJSONTag, ok := tags.LookupJSON(*context.Member)
	if !ok || fieldJSONTag.Name == "" {
		return Validations{}, fmt.Errorf("field %q has no JSON name", context.Member.Name)
	}
	fieldJSONName := fieldJSONTag.Name

	// Both fields must be directly comparable (==) and have the same
	// underlying type after resolving pointers and aliases.
	fieldBaseType := util.NonPointer(util.NativeType(context.Member.Type))
	siblingBaseType := util.NonPointer(util.NativeType(siblingMember.Type))

	if !util.IsDirectComparable(fieldBaseType) {
		return Validations{}, fmt.Errorf("field %q has non-comparable type %s", context.Member.Name, rootTypeString(context.Member.Type, fieldBaseType))
	}
	if !util.IsDirectComparable(siblingBaseType) {
		return Validations{}, fmt.Errorf("sibling field %q has non-comparable type %s", siblingMember.Name, rootTypeString(siblingMember.Type, siblingBaseType))
	}
	if fieldBaseType != siblingBaseType {
		return Validations{}, fmt.Errorf("type mismatch: field %q is %s but sibling %q is %s",
			context.Member.Name, fieldBaseType, siblingMember.Name, siblingBaseType)
	}

	ptrType := types.PointerTo(context.ParentType)
	fieldExtractor := createValueExtractor(ptrType, context.Member, fieldBaseType)
	siblingExtractor := createValueExtractor(ptrType, siblingMember, siblingBaseType)

	// Emit at the parent — the check needs both sibling fields.
	return Validations{
		Deferred: []DeferredGen{
			Deferred(ParentContext, func() (Validations, error) {
				fn := Function(equalToTagName, DefaultFlags, equalToValidator,
					fieldJSONName, fieldExtractor,
					siblingJSONName, siblingExtractor,
				).WithEmits(Emission{
					Type:         field.ErrorTypeInvalid,
					Origin:       "equalTo",
					PathFragment: "." + fieldJSONName,
				})
				return Validations{Functions: []FunctionGen{fn}}, nil
			}),
		},
	}, nil
}

// createValueExtractor builds a FunctionLiteral that extracts a field's
// comparable value from the parent struct pointer. The returned function has
// the signature func(obj *ParentType) ValueType.
func createValueExtractor(ptrType *types.Type, member *types.Member, valueType *types.Type) FunctionLiteral {
	nt := util.NativeType(member.Type)
	extractor := FunctionLiteral{
		Parameters: []ParamResult{{Name: "obj", Type: ptrType}},
		Results:    []ParamResult{{Type: valueType}},
	}
	switch nt.Kind {
	case types.Pointer:
		// Dereference pointer; return zero value if nil.
		extractor.Body = fmt.Sprintf("if obj == nil || obj.%s == nil {var z %s; return z}; return *obj.%s",
			member.Name, valueType, member.Name)
	default:
		// Direct value (builtin, struct, array).
		extractor.Body = fmt.Sprintf("if obj == nil {var z %s; return z}; return obj.%s",
			valueType, member.Name)
	}
	return extractor
}

func (v equalToTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            v.TagName(),
		StabilityLevel: TagStabilityLevelAlpha,
		Scopes:         sets.List(v.ValidScopes()),
		Description:    "Verifies that this field's value equals a named sibling field's value.",
		Docs: "Both fields must be directly comparable (string, integer, boolean, " +
			"or struct composed entirely of comparable fields) and have the same underlying type. " +
			"The error is reported on the tagged field. One-directional: tag the field that " +
			"should report the error if values differ.",
		Args: []TagArgDoc{{
			Description: "<sibling-field-json-name>",
			Type:        codetags.ArgTypeString,
			Required:    true,
			Docs:        "The JSON name of the sibling field that this field must equal.",
		}},
	}
}
