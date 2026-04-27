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
	"slices"

	"k8s.io/apimachinery/pkg/api/validate"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	updateTagName = "k8s:update"
)

func init() {
	shared := map[string]*updateMetadata{}
	RegisterFieldValidator(updateFieldValidator{byFieldPath: shared, listByPath: globalListMeta})
	RegisterTagValidator(updateTagCollector{byFieldPath: shared, listByPath: globalListMeta})
}

// updateMetadata collects constraints for a field, supporting both normal and shadow validation.
type updateMetadata struct {
	constraints    sets.Set[validate.UpdateConstraint]
	stabilityLevel ValidationStabilityLevel
}

// updateTagCollector collects +k8s:update tags
type updateTagCollector struct {
	byFieldPath map[string]*updateMetadata
	listByPath  map[string]*listMetadata
}

func (updateTagCollector) Init(_ Config) {}

func (updateTagCollector) TagName() string {
	return updateTagName
}

var updateTagValidScopes = sets.New(ScopeField, ScopeListVal, ScopeMapVal)

func (updateTagCollector) ValidScopes() sets.Set[Scope] {
	return updateTagValidScopes
}

func (utc updateTagCollector) GetValidations(context Context, tag codetags.Tag) (Validations, error) {
	// Parse constraint from this tag
	var constraint validate.UpdateConstraint
	switch tag.Value {
	case "NoSet":
		constraint = validate.NoSet
	case "NoUnset":
		constraint = validate.NoUnset
	case "NoModify":
		constraint = validate.NoModify
	case "NoAddItem":
		constraint = validate.NoAddItem
	case "NoRemoveItem":
		constraint = validate.NoRemoveItem
	default:
		return Validations{}, fmt.Errorf("unknown +k8s:update constraint: %s", tag.Value)
	}

	// Element scope (reached via +k8s:eachVal): only NoModify is valid,
	// and there is at most one constraint per tag invocation, so emit the
	// validation directly.
	// Field scope: fall through and accumulate, so that multiple +k8s:update
	// tags on the same field can be merged into one call later by
	// updateFieldValidator.
	if context.Scope == ScopeListVal || context.Scope == ScopeMapVal {
		if constraint != validate.NoModify {
			return Validations{}, fmt.Errorf("+k8s:update=%s does not apply to %s, attach it to the enclosing field", constraintName(constraint), context.Scope)
		}
		nt := util.NonPointer(util.NativeType(context.Type))
		if nt.Kind == types.Slice || nt.Kind == types.Map {
			return Validations{}, fmt.Errorf("+k8s:update=NoModify cannot be applied to list/map elements that are themselves lists or maps")
		}
		// For ScopeListVal, NoModify is only meaningful when the enclosing
		// list matches items by key (listType=map or unique=map). For
		// listType=set or listType=atomic, a content change becomes a new
		// unmatched item that ratchets through as a no-op. Reject upfront.
		if context.Scope == ScopeListVal && context.ParentPath != nil {
			if lm := utc.listByPath[context.ParentPath.String()]; lm != nil && lm.semantic != semanticMap {
				return Validations{}, fmt.Errorf("+k8s:eachVal=+k8s:update=NoModify requires the enclosing list to use listType=map or unique=map (got %s)", lm.semantic)
			}
		}
		v := emitScalarUpdate(context, []validate.UpdateConstraint{constraint})
		applyStabilityLevel(&v, context.StabilityLevel)
		return v, nil
	}

	// Initialize metadata if doesn't exist
	fieldPath := context.Path.String()
	if utc.byFieldPath[fieldPath] == nil {
		utc.byFieldPath[fieldPath] = &updateMetadata{constraints: sets.New[validate.UpdateConstraint]()}
	}
	um := utc.byFieldPath[fieldPath]

	um.stabilityLevel = context.StabilityLevel

	// Add this constraint to the set for this field
	um.constraints.Insert(constraint)

	if err := utc.validateConstraintsForType(context, um.constraints.UnsortedList()); err != nil {
		return Validations{}, err
	}

	// Don't generate validations here, just collect
	return Validations{}, nil
}

func (utc updateTagCollector) validateConstraintsForType(context Context, constraints []validate.UpdateConstraint) error {
	t := util.NonPointer(util.NativeType(context.Type))
	isCompound := t.Kind == types.Slice || t.Kind == types.Map
	isPointer := context.Type.Kind == types.Pointer
	isStruct := t.Kind == types.Struct

	for _, constraint := range constraints {
		switch constraint {
		case validate.NoAddItem, validate.NoRemoveItem:
			if !isCompound {
				return fmt.Errorf("+k8s:update=%s can only be used on list or map fields", constraintName(constraint))
			}
		case validate.NoModify:
			if isCompound {
				return fmt.Errorf("+k8s:update=NoModify is not supported on list or map fields, use +k8s:eachVal=+k8s:update=NoModify for per-item immutability")
			}
		case validate.NoSet, validate.NoUnset:
			// For non-pointer struct fields, only NoModify is applicable
			if isStruct && !isPointer {
				return fmt.Errorf("+k8s:update=%s cannot be used on non-pointer struct fields (they cannot be unset)", constraintName(constraint))
			}
		}
	}

	return nil
}

func constraintName(c validate.UpdateConstraint) string {
	switch c {
	case validate.NoSet:
		return "NoSet"
	case validate.NoUnset:
		return "NoUnset"
	case validate.NoModify:
		return "NoModify"
	case validate.NoAddItem:
		return "NoAddItem"
	case validate.NoRemoveItem:
		return "NoRemoveItem"
	default:
		return fmt.Sprintf("Unknown(%d)", c)
	}
}

func (utc updateTagCollector) Docs() TagDoc {
	return TagDoc{
		Tag:            utc.TagName(),
		StabilityLevel: TagStabilityLevelBeta,
		Scopes:         sets.List(utc.ValidScopes()),
		PayloadsType:   codetags.ValueTypeString,
		Description: "Provides constraints on the allowed update operations of a field. " +
			"Constraints: NoSet (prevents unset->set transitions), NoUnset (prevents set->unset transitions), " +
			"NoModify (prevents value changes but allows set/unset transitions), " +
			"NoAddItem (prevents adding items to a slice or map), NoRemoveItem (prevents removing items from a slice or map). " +
			"Multiple constraints can be specified using multiple tags. " +
			"For non-pointer structs, NoSet and NoUnset have no effect as these fields cannot be unset. " +
			"For slice and map fields, 'unset' means len == 0. Slice item identity for NoAddItem/NoRemoveItem comes from " +
			"+k8s:listType/+k8s:listMapKey/+k8s:unique, for maps the key is the item identity. " +
			"NoModify is not supported on slices or maps, use +k8s:eachVal=+k8s:update=NoModify for per-item immutability. " +
			"On lists, +k8s:eachVal=+k8s:update=NoModify requires listType=map or unique=map, otherwise content changes are not detectable. " +
			"Examples: +k8s:update=NoModify +k8s:update=NoUnset for set-once fields, " +
			"+k8s:update=NoSet for fields that must be set at creation or never, " +
			"+k8s:update=NoAddItem +k8s:update=NoRemoveItem on a listType=map field to freeze the structural shape of the list.",
	}
}

// updateFieldValidator processes all collected update tags and generates validations
type updateFieldValidator struct {
	byFieldPath map[string]*updateMetadata
	listByPath  map[string]*listMetadata
}

func (updateFieldValidator) Init(_ Config) {}

func (updateFieldValidator) Name() string {
	return "updateFieldValidator"
}

var (
	updateValueValidator          = types.Name{Package: libValidationPkg, Name: "UpdateValueByCompare"}
	updatePointerValidator        = types.Name{Package: libValidationPkg, Name: "UpdatePointer"}
	updateValueByReflectValidator = types.Name{Package: libValidationPkg, Name: "UpdateValueByReflect"}
	updateStructValidator         = types.Name{Package: libValidationPkg, Name: "UpdateStruct"}
	updateSliceValidator          = types.Name{Package: libValidationPkg, Name: "UpdateSlice"}
	updateMapValidator            = types.Name{Package: libValidationPkg, Name: "UpdateMap"}

	// Constraint constants that will be used as arguments
	noSetConstraint        = types.Name{Package: libValidationPkg, Name: "NoSet"}
	noUnsetConstraint      = types.Name{Package: libValidationPkg, Name: "NoUnset"}
	noModifyConstraint     = types.Name{Package: libValidationPkg, Name: "NoModify"}
	noAddItemConstraint    = types.Name{Package: libValidationPkg, Name: "NoAddItem"}
	noRemoveItemConstraint = types.Name{Package: libValidationPkg, Name: "NoRemoveItem"}
)

func (ufv updateFieldValidator) GetValidations(context Context) (Validations, error) {
	um := ufv.byFieldPath[context.Path.String()]

	if um == nil || um.constraints.Len() == 0 {
		return Validations{}, nil
	}

	constraints := um.constraints.UnsortedList()

	v, err := ufv.generateValidation(context, constraints)
	if err != nil {
		return Validations{}, err
	}

	level := um.stabilityLevel
	if context.StabilityLevel != "" {
		level = context.StabilityLevel
	}

	applyStabilityLevel(&v, level)
	return v, nil
}

func (ufv updateFieldValidator) generateValidation(context Context, constraints []validate.UpdateConstraint) (Validations, error) {
	// Sort constraints to ensure deterministic order
	slices.Sort(constraints)

	t := util.NonPointer(util.NativeType(context.Type))
	switch t.Kind {
	case types.Slice:
		return ufv.generateSliceValidation(context, constraints)
	case types.Map:
		return generateMapValidation(constraints), nil
	}
	return emitScalarUpdate(context, constraints), nil
}

// generateSliceValidation emits validate.UpdateSlice. NoAddItem/NoRemoveItem
// need a match function to pair old and new items, which function we emit is
// determined by the list's semantic (set/map) as set by
// +k8s:listType/+k8s:listMapKey/+k8s:unique. For NoSet/NoUnset alone, a nil match is fine.
//
// Metadata lookup falls back from the field path to the type path so that
// typedef-level list annotations apply, mirroring the pattern used by
// listValidator.
func (ufv updateFieldValidator) generateSliceValidation(context Context, constraints []validate.UpdateConstraint) (Validations, error) {
	var matchArg any = Literal("nil")
	// NoAddItem/NoRemoveItem need a match function to pair items between old and new, NoSet/NoUnset only check len == 0.
	if slices.Contains(constraints, validate.NoAddItem) || slices.Contains(constraints, validate.NoRemoveItem) {
		lm := ufv.listByPath[context.Path.String()]
		if lm == nil {
			lm = ufv.listByPath[context.Type.String()]
		}
		if lm == nil {
			return Validations{}, fmt.Errorf("+k8s:update=NoAddItem/+k8s:update=NoRemoveItem require list metadata (+k8s:listType with listMapKey, or +k8s:unique) to determine item identity")
		}
		elem := util.NativeType(context.Type).Elem
		switch lm.semantic {
		case semanticMap:
			if len(lm.keyMembers) == 0 {
				return Validations{}, fmt.Errorf("+k8s:update=NoAddItem/+k8s:update=NoRemoveItem require listMapKey to be set for listType=map")
			}
			matchArg = lm.makeListMapMatchFunc(elem)
		case semanticSet:
			if util.IsDirectComparable(util.NonPointer(util.NativeType(elem))) {
				matchArg = Identifier(validateDirectEqual)
			} else {
				matchArg = Identifier(validateSemanticDeepEqual)
			}
		default:
			return Validations{}, fmt.Errorf("+k8s:update=NoAddItem/+k8s:update=NoRemoveItem require listType=set, listType=map, unique=set, or unique=map to define item identity")
		}
	}

	args := append([]any{matchArg}, constraintIdentifierArgs(constraints)...)

	// Use ShortCircuit flag so these run in the same group as +k8s:optional
	fn := Function(updateTagName, ShortCircuit, updateSliceValidator, args...)
	return Validations{Functions: []FunctionGen{fn}}, nil
}

// generateMapValidation emits validate.UpdateMap. The map key is the item
// identity, so no list metadata or match function is needed.
func generateMapValidation(constraints []validate.UpdateConstraint) Validations {
	// Use ShortCircuit flag so these run in the same group as +k8s:optional
	fn := Function(updateTagName, ShortCircuit, updateMapValidator, constraintIdentifierArgs(constraints)...)
	return Validations{Functions: []FunctionGen{fn}}
}

// emitScalarUpdate emits a call to
// UpdateValueByCompare/UpdatePointer/UpdateStruct/UpdateValueByReflect based
// on the Go kind of context.Type.
// Used for both field scope (scalar/pointer/struct) and list/map element
// scope via +k8s:eachVal.
func emitScalarUpdate(context Context, constraints []validate.UpdateConstraint) Validations {
	// Determine the appropriate validator function based on field type
	t := util.NonPointer(util.NativeType(context.Type))
	isPointer := context.Type.Kind == types.Pointer
	isStruct := t.Kind == types.Struct
	isComparable := util.IsDirectComparable(t)

	var validatorFunc types.Name
	if isPointer {
		validatorFunc = updatePointerValidator
	} else if isStruct {
		validatorFunc = updateStructValidator
	} else if isComparable {
		validatorFunc = updateValueValidator
	} else {
		validatorFunc = updateValueByReflectValidator
	}

	// Use ShortCircuit flag so these run in the same group as +k8s:optional
	fn := Function(updateTagName, ShortCircuit, validatorFunc, constraintIdentifierArgs(constraints)...)
	return Validations{Functions: []FunctionGen{fn}}
}

// constraintIdentifierArgs builds the constraint arguments in deterministic order.
func constraintIdentifierArgs(constraints []validate.UpdateConstraint) []any {
	var args []any
	for _, constraint := range constraints {
		switch constraint {
		case validate.NoSet:
			args = append(args, Identifier(noSetConstraint))
		case validate.NoUnset:
			args = append(args, Identifier(noUnsetConstraint))
		case validate.NoModify:
			args = append(args, Identifier(noModifyConstraint))
		case validate.NoAddItem:
			args = append(args, Identifier(noAddItemConstraint))
		case validate.NoRemoveItem:
			args = append(args, Identifier(noRemoveItemConstraint))
		}
	}
	return args
}

func applyStabilityLevel(v *Validations, level ValidationStabilityLevel) {
	if level == "" {
		return
	}
	for i := range v.Functions {
		v.Functions[i] = v.Functions[i].WithStabilityLevel(level)
	}
}
