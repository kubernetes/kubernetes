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

var zeroOrOneOfUnionValidator = types.Name{Package: libValidationPkg, Name: "ZeroOrOneOfUnion"}
var zeroOrOneOfVariablePrefix = "zeroOrOneOfMembershipFor"

func init() {
	// ZeroOrOneOf unions are comprised of multiple tags, which need to share information
	// between them.  The tags are on struct fields, but the validation
	// actually pertains to the struct itself.
	shared := map[string]unions{}
	RegisterTypeValidator(zeroOrOneOfTypeOrFieldValidator{shared})
	RegisterFieldValidator(zeroOrOneOfTypeOrFieldValidator{shared})
	RegisterTagValidator(zeroOrOneOfMemberTagValidator{shared})
}

type zeroOrOneOfTypeOrFieldValidator struct {
	shared map[string]unions
}

func (zeroOrOneOfTypeOrFieldValidator) Init(_ Config) {}

func (zeroOrOneOfTypeOrFieldValidator) Name() string {
	return "zeroOrOneOfTypeOrFieldValidator"
}

func (ztfv zeroOrOneOfTypeOrFieldValidator) GetValidations(context Context) (Validations, error) {
	// Gengo does not treat struct definitions as aliases, which is
	// inconsistent but unlikely to change. That means we don't REALLY need to
	// handle it here, but let's be extra careful and extract the most concrete
	// type possible.
	if k := util.NonPointer(util.NativeType(context.Type)).Kind; k != types.Struct && k != types.Slice {
		return Validations{}, nil
	}

	unions := ztfv.shared[context.Path.String()]
	if len(unions) == 0 {
		return Validations{}, nil
	}

	return processUnionValidations(context, unions, zeroOrOneOfVariablePrefix,
		zeroOrOneOfMemberTagName, zeroOrOneOfUnionValidator, types.Name{})
}

const (
	// This tag should only ever be used on list item types, never on struct
	// fields directly.  If applied to struct fields, the "orR one of" behavior
	// is frozen at this moment in time, and can never be expanded. Why?
	// Back-rev clients can't tell the difference between "zero were specified"
	// and "a field I don't know about was specified".
	zeroOrOneOfMemberTagName = "k8s:zeroOrOneOfMember"
)

type zeroOrOneOfMemberTagValidator struct {
	shared map[string]unions
}

func (zeroOrOneOfMemberTagValidator) Init(_ Config) {}

func (zeroOrOneOfMemberTagValidator) TagName() string {
	return zeroOrOneOfMemberTagName
}

func (zeroOrOneOfMemberTagValidator) ValidScopes() sets.Set[Scope] {
	return sets.New(ScopeField, ScopeListVal)
}

func (zmtv zeroOrOneOfMemberTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	err := processMemberValidations(zmtv.shared, context, tag)
	if err != nil {
		return Validations{}, err
	}
	// This tag does not actually emit any validations, it just accumulates
	// information. The validation is done by the zeroOrOneOfTypeOrFieldValidator.
	return Validations{}, nil
}

func (zmtv zeroOrOneOfMemberTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            zmtv.TagName(),
		Scopes:         zmtv.ValidScopes().UnsortedList(),
		StabilityLevel: Beta,
		Description:    "Indicates that this field is a member of a zero-or-one-of union.",
		Docs:           "A zero-or-one-of union allows at most one member to be set. Unlike regular unions, having no members set is valid.",
		Warning:        "This tag should only be used on sets of list items, and never on struct fields directly.",
		Args: []TagArgDoc{{
			Name:        "union",
			Description: "<string>",
			Docs:        "the name of the union, if more than one exists",
			Type:        codetags.ArgTypeString,
		}, {
			Name:        "memberName",
			Description: "<string>",
			Docs:        "the custom member name for this member",
			Default:     "the field's name",
			Type:        codetags.ArgTypeString,
		}},
	}
}
