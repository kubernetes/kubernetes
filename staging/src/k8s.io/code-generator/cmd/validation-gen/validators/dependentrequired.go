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
	dependentRequiredTagName = "k8s:dependentRequired"
)

var dependentRequiredValidator = types.Name{Package: libValidationPkg, Name: "DependentRequired"}

func init() {
	RegisterTagValidator(dependentRequiredTagValidator{})
}

type dependentRequiredTagValidator struct{}

func (dependentRequiredTagValidator) Init(_ Config) {}

func (dependentRequiredTagValidator) TagName() string {
	return dependentRequiredTagName
}

var dependentRequiredTagValidScopes = sets.New(ScopeField)

func (dependentRequiredTagValidator) ValidScopes() sets.Set[Scope] {
	return dependentRequiredTagValidScopes
}

func (drv dependentRequiredTagValidator) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	if context.Member == nil {
		return Validations{}, fmt.Errorf("must be used on a struct field")
	}

	parentType := util.NonPointer(util.NativeType(context.ParentType))
	if parentType.Kind != types.Struct {
		return Validations{}, fmt.Errorf("parent type must be a struct (got %s)", parentType.Kind)
	}

	if len(tag.Args) != 1 {
		return Validations{}, fmt.Errorf("expected exactly one argument naming the dependent field, got %d", len(tag.Args))
	}
	dependentJSONName := tag.Args[0].Value

	dependentMember := util.GetMemberByJSON(parentType, dependentJSONName)
	if dependentMember == nil {
		return Validations{}, fmt.Errorf("no sibling field with JSON name %q", dependentJSONName)
	}
	if dependentMember.Name == context.Member.Name {
		return Validations{}, fmt.Errorf("trigger and dependent field must be different (both are %q)", dependentJSONName)
	}

	triggerJSONTag, ok := tags.LookupJSON(*context.Member)
	if !ok || triggerJSONTag.Name == "" {
		return Validations{}, fmt.Errorf("trigger field %q has no JSON name", context.Member.Name)
	}
	triggerJSONName := triggerJSONTag.Name

	// "Is set" has an obvious meaning for these kinds: pointer non-nil,
	// slice/map non-empty, builtin non-zero. We don't restrict what a
	// pointer points to — in practice it's a builtin or a struct.
	for _, m := range []struct {
		role   string
		member *types.Member
	}{{"trigger", context.Member}, {"dependent", dependentMember}} {
		nt := util.NativeType(m.member.Type)
		switch nt.Kind {
		case types.Pointer, types.Map, types.Slice, types.Builtin:
		default:
			return Validations{}, fmt.Errorf("%s field %q has unsupported type kind %s",
				m.role, m.member.Name, nt.Kind)
		}
	}

	ptrType := types.PointerTo(context.ParentType)
	triggerExtractor := createMemberExtractor(ptrType, context.Member)
	dependentExtractor := createMemberExtractor(ptrType, dependentMember)

	// Emit at the parent — the check needs both sibling fields.
	return Validations{
		Deferred: []DeferredGen{
			Deferred(ParentContext, func() (Validations, error) {
				fn := Function(dependentRequiredTagName, DefaultFlags, dependentRequiredValidator,
					triggerJSONName, triggerExtractor,
					dependentJSONName, dependentExtractor,
				).WithEmits(Emission{
					Type:         field.ErrorTypeRequired,
					Origin:       "dependentRequired",
					PathFragment: "." + dependentJSONName,
				})
				return Validations{Functions: []FunctionGen{fn}}, nil
			}),
		},
	}, nil
}

func (drv dependentRequiredTagValidator) Docs() TagDoc {
	return TagDoc{
		Tag:            drv.TagName(),
		StabilityLevel: TagStabilityLevelAlpha,
		Scopes:         sets.List(drv.ValidScopes()),
		Description:    "Indicates that when this field is set, the named sibling field must also be set.",
		Args: []TagArgDoc{{
			Description: "<sibling-field-json-name>",
			Type:        codetags.ArgTypeString,
			Required:    true,
		}},
		Docs: "When the tagged field is set (non-nil pointer, non-empty slice/map, or non-zero " +
			"builtin), the named sibling must also be set. Dependencies are one-directional. " +
			"Repeat the tag to require multiple siblings.",
	}
}
