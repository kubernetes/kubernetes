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
	"sort"

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
	shared := map[string]sets.Set[validate.UpdateConstraint]{}
	RegisterFieldValidator(updateFieldValidator{byFieldPath: shared})
	RegisterTagValidator(updateTagCollector{byFieldPath: shared})
}

// updateTagCollector collects +k8s:update tags
type updateTagCollector struct {
	byFieldPath map[string]sets.Set[validate.UpdateConstraint]
}

func (updateTagCollector) Init(_ Config) {}

func (updateTagCollector) TagName() string {
	return updateTagName
}

var updateTagValidScopes = sets.New(ScopeField)

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
	default:
		return Validations{}, fmt.Errorf("unknown +k8s:update constraint: %s", tag.Value)
	}

	// Initialize set if doesn't exist
	fieldPath := context.Path.String()
	if utc.byFieldPath[fieldPath] == nil {
		utc.byFieldPath[fieldPath] = sets.New[validate.UpdateConstraint]()
	}

	// Add this constraint to the set for this field
	utc.byFieldPath[fieldPath].Insert(constraint)

	if err := utc.validateConstraintsForType(context, utc.byFieldPath[fieldPath].UnsortedList()); err != nil {
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

	if isCompound {
		for _, constraint := range constraints {
			return fmt.Errorf("+k8s:update=%s is currently not supported on list or map fields", constraintName(constraint))
		}
	}

	// For non-pointer struct fields, only NoModify is applicable
	if isStruct && !isPointer {
		for _, constraint := range constraints {
			if constraint == validate.NoSet || constraint == validate.NoUnset {
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
	default:
		return fmt.Sprintf("Unknown(%d)", c)
	}
}

func (utc updateTagCollector) Docs() TagDoc {
	return TagDoc{
		Tag:            utc.TagName(),
		StabilityLevel: Alpha,
		Scopes:         utc.ValidScopes().UnsortedList(),
		PayloadsType:   codetags.ValueTypeString,
		Description: "Provides constraints on the allowed update operations of a field. " +
			"Currently supports non-list and non-map fields only. " +
			"Constraints: NoSet (prevents unset->set transitions), NoUnset (prevents set->unset transitions), " +
			"NoModify (prevents value changes but allows set/unset transitions). " +
			"Multiple constraints can be specified using multiple tags. " +
			"For non-pointer structs, NoSet and NoUnset have no effect as these fields cannot be unset. " +
			"Future support planned for lists/maps with NoAddItem and NoRemoveItem constraints. " +
			"Examples: +k8s:update=NoModify +k8s:update=NoUnset for set-once fields; " +
			"+k8s:update=NoSet for fields that must be set at creation or never.",
	}
}

// updateFieldValidator processes all collected update tags and generates validations
type updateFieldValidator struct {
	byFieldPath map[string]sets.Set[validate.UpdateConstraint]
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

	// Constraint constants that will be used as arguments
	noSetConstraint    = types.Name{Package: libValidationPkg, Name: "NoSet"}
	noUnsetConstraint  = types.Name{Package: libValidationPkg, Name: "NoUnset"}
	noModifyConstraint = types.Name{Package: libValidationPkg, Name: "NoModify"}
)

func (ufv updateFieldValidator) GetValidations(context Context) (Validations, error) {
	constraintSet, ok := ufv.byFieldPath[context.Path.String()]
	if !ok || constraintSet.Len() == 0 {
		return Validations{}, nil
	}

	constraints := constraintSet.UnsortedList()

	t := util.NonPointer(util.NativeType(context.Type))
	if t.Kind == types.Slice || t.Kind == types.Map {
		// TODO: add support for list and map fields
		return Validations{}, fmt.Errorf("update constraints are currently not supported on list or map fields")
	}

	return ufv.generateValidation(context, constraints)
}

func (ufv updateFieldValidator) generateValidation(context Context, constraints []validate.UpdateConstraint) (Validations, error) {
	var result Validations

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

	// Sort constraints to ensure deterministic order
	sort.Slice(constraints, func(i, j int) bool {
		return constraints[i] < constraints[j]
	})

	// Build the constraint arguments in deterministic order
	var constraintArgs []any
	for _, constraint := range constraints {
		switch constraint {
		case validate.NoSet:
			constraintArgs = append(constraintArgs, Identifier(noSetConstraint))
		case validate.NoUnset:
			constraintArgs = append(constraintArgs, Identifier(noUnsetConstraint))
		case validate.NoModify:
			constraintArgs = append(constraintArgs, Identifier(noModifyConstraint))
		}
	}

	// Use ShortCircuit flag so these run in the same group as +k8s:optional
	fn := Function(updateTagName, ShortCircuit, validatorFunc, constraintArgs...)
	result.AddFunction(fn)

	return result, nil
}
