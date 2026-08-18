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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/parser/tags"
	"k8s.io/gengo/v2/types"
)

const (
	dependentRequiredTagName  = "k8s:dependentRequired"
	dependentForbiddenTagName = "k8s:dependentForbidden"
)

var (
	dependentRequiredValidator  = types.Name{Package: libValidationPkg, Name: "DependentRequired"}
	dependentForbiddenValidator = types.Name{Package: libValidationPkg, Name: "DependentForbidden"}
)

func init() {
	RegisterTagValidator(dependencyTagValidator{dependencyRequired})
	RegisterTagValidator(dependencyTagValidator{dependencyForbidden})
}

// dependencyTagValidator implements conditional set-ness dependencies between
// sibling fields: when the tagged (trigger) field is set, a named sibling must
// also be set (required mode) or must not be set (forbidden mode).
type dependencyTagValidator struct {
	mode dependencyMode
}

type dependencyMode string

const (
	dependencyRequired  dependencyMode = dependentRequiredTagName
	dependencyForbidden dependencyMode = dependentForbiddenTagName
)

type dependencyMetadata struct {
	mode              dependencyMode
	triggerMember     *types.Member
	dependentMember   *types.Member
	triggerJSONName   string
	dependentJSONName string
	stabilityLevel    ValidationStabilityLevel
	conditions        Conditions
}

func (dm dependencyMetadata) Compare(other dependencyMetadata) int {
	if cmp := strings.Compare(dm.triggerJSONName, other.triggerJSONName); cmp != 0 {
		return cmp
	}
	if cmp := strings.Compare(dm.dependentJSONName, other.dependentJSONName); cmp != 0 {
		return cmp
	}
	if cmp := strings.Compare(string(dm.mode), string(other.mode)); cmp != 0 {
		return cmp
	}
	return dm.conditions.Compare(other.conditions)
}

func (dm dependencyMetadata) DeepCopy() dependencyMetadata {
	return dm
}

func (dependencyTagValidator) Init(_ Config) {}

func (dtv dependencyTagValidator) TagName() string {
	return string(dtv.mode)
}

var dependencyTagValidScopes = sets.New(ScopeType, ScopeField)

func (dependencyTagValidator) ValidScopes() sets.Set[Scope] {
	return dependencyTagValidScopes
}

func (dtv dependencyTagValidator) CollectMetadata(context Context, tag codetags.Tag) (SchemaMetadata, error) {
	if context.Member == nil {
		return SchemaMetadata{}, fmt.Errorf("must be used on a struct field")
	}

	parentType := util.NonPointer(util.NativeType(context.ParentType))
	if parentType.Kind != types.Struct {
		return SchemaMetadata{}, fmt.Errorf("parent type must be a struct (got %s)", parentType.Kind)
	}

	if len(tag.Args) != 1 {
		return SchemaMetadata{}, fmt.Errorf("expected exactly one argument naming the dependent field, got %d", len(tag.Args))
	}
	dependentJSONName := tag.Args[0].Value

	dependentMember := util.GetMemberByJSON(parentType, dependentJSONName)
	if dependentMember == nil {
		return SchemaMetadata{}, fmt.Errorf("no sibling field with JSON name %q", dependentJSONName)
	}
	if dependentMember.Name == context.Member.Name {
		return SchemaMetadata{}, fmt.Errorf("trigger and dependent field must be different (both are %q)", dependentJSONName)
	}

	triggerJSONTag, ok := tags.LookupJSON(*context.Member)
	if !ok || triggerJSONTag.Name == "" {
		return SchemaMetadata{}, fmt.Errorf("trigger field %q has no JSON name", context.Member.Name)
	}
	triggerJSONName := triggerJSONTag.Name

	for _, m := range []struct {
		role   string
		member *types.Member
	}{{"trigger", context.Member}, {"dependent", dependentMember}} {
		nt := util.NativeType(m.member.Type)
		switch nt.Kind {
		case types.Pointer, types.Map, types.Slice, types.Builtin:
		default:
			return SchemaMetadata{}, fmt.Errorf("%s field %q has unsupported type kind %s",
				m.role, m.member.Name, nt.Kind)
		}
	}

	meta := dependencyMetadata{
		mode:              dtv.mode,
		triggerMember:     context.Member,
		dependentMember:   dependentMember,
		triggerJSONName:   triggerJSONName,
		dependentJSONName: dependentJSONName,
		stabilityLevel:    context.StabilityLevel,
		conditions:        context.Conditions,
	}

	res := SchemaMetadata{}
	node := res.GetOrCreateNode(nodeKeyFor(context.Path))
	node.Dependencies = []Conditional[*dependencyMetadata]{
		{
			Conditions:     context.Conditions,
			StabilityLevel: context.StabilityLevel,
			Payload:        &meta,
		},
	}
	return res, nil
}

func (dtv dependencyTagValidator) GetValidations(context Context, metadata SchemaMetadata, tag codetags.Tag) (Validations, error) {
	if context.Scope != ScopeField {
		return Validations{}, nil
	}
	deps := metadata.SortedDependencies()
	if len(deps) == 0 {
		return Validations{}, nil
	}

	hasUnemitted := false
	for _, dm := range deps {
		emittedKey := fmt.Sprintf("dep:%s:%s:%s:%s", dm.mode, dm.triggerJSONName, dm.dependentJSONName, dm.conditions.Key())
		if !metadata.MarkEmitted(emittedKey) {
			hasUnemitted = true
		}
	}
	if !hasUnemitted {
		return Validations{}, nil
	}

	structType := context.ParentType
	if structType == nil {
		return Validations{}, fmt.Errorf("missing ParentType for field scope")
	}
	if k := util.NonPointer(util.NativeType(structType)).Kind; k != types.Struct {
		return Validations{}, nil
	}

	var result Validations
	ptrType := types.PointerTo(structType)

	for _, dm := range deps {
		cond := dm.conditions
		var validator types.Name
		var emitType field.ErrorType
		var origin string
		switch dm.mode {
		case dependencyRequired:
			validator, emitType, origin = dependentRequiredValidator, field.ErrorTypeRequired, "dependentRequired"
		case dependencyForbidden:
			validator, emitType, origin = dependentForbiddenValidator, field.ErrorTypeForbidden, "dependentForbidden"
		default:
			return Validations{}, fmt.Errorf("unknown dependency mode: %q", dm.mode)
		}

		// Emit at the parent — the check needs both sibling fields.
		triggerExtractor := createMemberExtractor(ptrType, dm.triggerMember)
		dependentExtractor := createMemberExtractor(ptrType, dm.dependentMember)

		fn := Function(string(dm.mode), DefaultFlags, validator,
			dm.triggerJSONName, triggerExtractor,
			dm.dependentJSONName, dependentExtractor,
		).WithEmits(Emission{
			Type:         emitType,
			Origin:       origin,
			PathFragment: "." + dm.dependentJSONName,
		})

		finalized, err := FinalizeGroup(context, EmittedGroup{
			Validations:    Validations{Functions: []FunctionGen{fn}},
			Conditions:     cond,
			StabilityLevel: dm.stabilityLevel,
			TargetType:     structType,
			Hoist:          true,
		})
		if err != nil {
			return Validations{}, err
		}
		result.Add(finalized)
	}

	return result, nil
}

func (dtv dependencyTagValidator) Docs() TagDoc {
	doc := TagDoc{
		Tag:            dtv.TagName(),
		StabilityLevel: TagStabilityLevelAlpha,
		Scopes:         sets.List(dtv.ValidScopes()),
		Args: []TagArgDoc{{
			Description: "<sibling-field-json-name>",
			Type:        codetags.ArgTypeString,
			Required:    true,
		}},
	}
	switch dtv.mode {
	case dependencyRequired:
		doc.Description = "Indicates that when this field is set, the named sibling field must also be set."
		doc.Docs = "When the tagged field is set (non-nil pointer, non-empty slice/map, or non-zero " +
			"builtin), the named sibling must also be set. Dependencies are one-directional. " +
			"Repeat the tag to require multiple siblings."
	case dependencyForbidden:
		doc.Description = "Indicates that when this field is set, the named sibling field must not be set."
		doc.Docs = "When the tagged field is set (non-nil pointer, non-empty slice/map, or non-zero " +
			"builtin), the named sibling must not be set. Dependencies are one-directional. " +
			"Repeat the tag to forbid multiple siblings."
	default:
		panic(fmt.Sprintf("unknown dependency mode: %q", dtv.mode))
	}
	return doc
}
