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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/code-generator/cmd/validation-gen/util"
	"k8s.io/gengo/v2/codetags"
	"k8s.io/gengo/v2/types"
)

const (
	updateTagName = "k8s:update"
)

type updateConstraint string

const (
	// constraintNoSet prevents unset->set transitions
	constraintNoSet updateConstraint = "NoSet"
	// constraintNoUnset prevents set->unset transitions
	constraintNoUnset updateConstraint = "NoUnset"
	// constraintNoModify prevents value changes but allows set/unset transitions
	constraintNoModify updateConstraint = "NoModify"
)

func init() {
	shared := map[string][]updateConstraint{}
	RegisterFieldValidator(updateFieldValidator{byFieldPath: shared})
	RegisterTagValidator(updateTagCollector{byFieldPath: shared})
}

// updateTagCollector collects +k8s:update tags
type updateTagCollector struct {
	byFieldPath map[string][]updateConstraint
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
	tagValue := strings.TrimSpace(tag.Value)
	tagValue = strings.Trim(tagValue, "`\"")

	if tagValue == "" {
		return Validations{}, nil
	}

	var constraints []updateConstraint

	// Parse constraints from payload
	for _, value := range strings.Split(tagValue, ",") {
		constraintStr := strings.TrimSpace(value)
		var constraint updateConstraint
		switch constraintStr {
		case string(constraintNoSet):
			constraint = constraintNoSet
		case string(constraintNoUnset):
			constraint = constraintNoUnset
		case string(constraintNoModify):
			constraint = constraintNoModify
		default:
			return Validations{}, fmt.Errorf("unknown +k8s:update constraint: %s", constraintStr)
		}

		constraints = append(constraints, constraint)
	}

	// Validate constraints are appropriate for the field type
	if err := utc.validateConstraintsForType(context, constraints); err != nil {
		return Validations{}, err
	}

	// Store the constraints for this field
	utc.byFieldPath[context.Path.String()] = constraints

	// Don't generate validations here, just collect
	return Validations{}, nil
}

func (utc updateTagCollector) validateConstraintsForType(context Context, constraints []updateConstraint) error {
	t := util.NonPointer(util.NativeType(context.Type))
	isCompound := t.Kind == types.Slice || t.Kind == types.Map
	isPointer := context.Type.Kind == types.Pointer
	isStruct := t.Kind == types.Struct

	if isCompound {
		for _, constraint := range constraints {
			return fmt.Errorf("+k8s:update=%s is currently not supported on list or map fields", constraint)
		}
	}

	// For non-pointer struct fields, only NoModify is applicable
	if isStruct && !isPointer {
		for _, constraint := range constraints {
			if constraint == constraintNoSet || constraint == constraintNoUnset {
				return fmt.Errorf("+k8s:update=%s cannot be used on non-pointer struct fields (they cannot be unset)", constraint)
			}
		}
	}

	return nil
}

func (utc updateTagCollector) Docs() TagDoc {
	return TagDoc{
		Tag:          utc.TagName(),
		Scopes:       utc.ValidScopes().UnsortedList(),
		PayloadsType: codetags.ValueTypeString,
		Description: "Provides fine-grained control over field update operations. " +
			"Only for non-list/map fields: NoSet (prevents unset->set), NoUnset (prevents set->unset), " +
			"NoModify (prevents value changes but allows set/unset transitions). " +
			"Multiple values can be specified separated by commas. " +
			"Examples: +k8s:update=`NoModify,NoUnset` for set-once fields; " +
			"+k8s:update=`NoSet` for fields that must be set at creation or never.",
	}
}

// updateFieldValidator processes all collected update tags and generates validations
type updateFieldValidator struct {
	byFieldPath map[string][]updateConstraint
}

func (updateFieldValidator) Init(_ Config) {}

func (updateFieldValidator) Name() string {
	return "updateFieldValidator"
}

var (
	noSetValueValidator             = types.Name{Package: libValidationPkg, Name: "NoSetValue"}
	noSetPointerValidator           = types.Name{Package: libValidationPkg, Name: "NoSetPointer"}
	noUnsetValueValidator           = types.Name{Package: libValidationPkg, Name: "NoUnsetValue"}
	noUnsetPointerValidator         = types.Name{Package: libValidationPkg, Name: "NoUnsetPointer"}
	noModifyValueValidator          = types.Name{Package: libValidationPkg, Name: "NoModifyValue"}
	noModifyValueByReflectValidator = types.Name{Package: libValidationPkg, Name: "NoModifyValueByReflect"}
	noModifyPointerValidator        = types.Name{Package: libValidationPkg, Name: "NoModifyPointer"}
	noModifyStructValidator         = types.Name{Package: libValidationPkg, Name: "NoModifyStruct"}
)

func (ufv updateFieldValidator) GetValidations(context Context) (Validations, error) {
	constraints, ok := ufv.byFieldPath[context.Path.String()]
	if !ok || len(constraints) == 0 {
		return Validations{}, nil
	}

	t := util.NonPointer(util.NativeType(context.Type))
	if t.Kind == types.Slice || t.Kind == types.Map {
		// TODO: add support for list and map fields
		return Validations{}, fmt.Errorf("update constraints are currently not supported on list or map fields")
	}

	return ufv.generateValidations(context, constraints)
}

func (ufv updateFieldValidator) generateValidations(context Context, constraints []updateConstraint) (Validations, error) {
	var result Validations

	// Determine the appropriate validator functions based on field type
	t := util.NonPointer(util.NativeType(context.Type))
	isPointer := context.Type.Kind == types.Pointer
	isStruct := t.Kind == types.Struct
	isComparable := util.IsDirectComparable(t)
	var noSetValidator, noUnsetValidator, noModifyValidator types.Name

	if isPointer {
		noSetValidator = noSetPointerValidator
		noUnsetValidator = noUnsetPointerValidator
		noModifyValidator = noModifyPointerValidator
	} else if isStruct {
		// Struct types can only use NoModify
		noModifyValidator = noModifyStructValidator
	} else {
		noSetValidator = noSetValueValidator
		noUnsetValidator = noUnsetValueValidator
		if isComparable {
			noModifyValidator = noModifyValueValidator
		} else {
			noModifyValidator = noModifyValueByReflectValidator
		}
	}

	// Use ShortCircuit flag so these run in the same group as +k8s:optional
	for _, constraint := range constraints {
		switch constraint {
		case constraintNoSet:
			if noSetValidator.Name != "" {
				result.AddFunction(Function("update:NoSet", ShortCircuit, noSetValidator))
			}
		case constraintNoUnset:
			if noUnsetValidator.Name != "" {
				result.AddFunction(Function("update:NoUnset", ShortCircuit, noUnsetValidator))
			}
		case constraintNoModify:
			if noModifyValidator.Name != "" {
				result.AddFunction(Function("update:NoModify", ShortCircuit, noModifyValidator))
			}
		}
	}

	return result, nil
}
