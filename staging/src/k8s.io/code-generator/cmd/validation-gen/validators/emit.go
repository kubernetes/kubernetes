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

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/gengo/v2/types"
)

// ValidationGroup is a batch of validations plus declarative facts about where
// and when they apply. Producers (e.g. the union or item generators) state
// facts; they never apply wrapping themselves. Each fact is consumed by the
// ValidationWrapper registered by the validator that owns that concern:
// Conditions by options.go, StabilityLevel by levels.go, FieldPath by
// subfield.go.
type ValidationGroup struct {
	Validations Validations

	// Conditions that must hold for these validations to run.
	Conditions Conditions

	// StabilityLevel to mark these validations with, if any.
	StabilityLevel ValidationStabilityLevel

	// FieldPath is the absolute schema path the validations belong to. When
	// it names a descendant of the emitting context's path, the validations
	// are re-homed there. Nil means the current context.
	FieldPath *field.Path

	// FieldType is the type at FieldPath, used as the object type for
	// generated wrapper functions. Nil means the current context's type.
	FieldType *types.Type

	// EmitOnType indicates the finalized functions belong on the enclosing type's
	// validation function: after the finalizer chain runs, they are moved to
	// TypeFunctions. Variables and comments are unaffected.
	EmitOnType bool
}

// ValidationWrapper materializes one cross-cutting concern of an
// ValidationGroup. A finalizer consumes its envelope field(s), clearing them
// when applied, and must be a no-op passthrough when its field is empty.
type ValidationWrapper interface {
	Name() string
	Wrap(context Context, group ValidationGroup) (ValidationGroup, error)
}

// Finalizer chain positions. Wrapping nests in chain order: an earlier
// finalizer's wrappers end up innermost in the generated code. The resulting
// nesting (conditions innermost, then stability marking, subfield re-homing
// outermost) is visible in generated output and must not change silently.
const (
	FinalizeConditionsOrder = 100
	FinalizeStabilityOrder  = 200
	FinalizeSubfieldOrder   = 300
)

type registeredFinalizer struct {
	order     int
	finalizer ValidationWrapper
}

var validationWrappers []registeredFinalizer

// RegisterValidationWrapper adds a finalizer to the emission chain at the
// given position. Positions must be unique so the chain order is unambiguous.
func RegisterValidationWrapper(order int, f ValidationWrapper) {
	for _, existing := range validationWrappers {
		if existing.order == order {
			panic(fmt.Sprintf("emission finalizer order %d already taken by %q, cannot register %q", order, existing.finalizer.Name(), f.Name()))
		}
	}
	validationWrappers = append(validationWrappers, registeredFinalizer{order: order, finalizer: f})
	slices.SortFunc(validationWrappers, func(a, b registeredFinalizer) int {
		return a.order - b.order
	})
}

// WrapGroup runs the emission finalizer chain over one group and returns
// the resulting validations.
func WrapGroup(context Context, group ValidationGroup) (Validations, error) {
	for _, rf := range validationWrappers {
		var err error
		group, err = rf.finalizer.Wrap(context, group)
		if err != nil {
			return Validations{}, fmt.Errorf("emission finalizer %q: %w", rf.finalizer.Name(), err)
		}
	}
	if group.EmitOnType {
		group.Validations.TypeFunctions = append(group.Validations.TypeFunctions, group.Validations.Functions...)
		group.Validations.Functions = nil
	}
	return group.Validations, nil
}

// WrapGroups runs the emission finalizer chain over each group and merges
// the results.
func WrapGroups(context Context, groups []ValidationGroup) (Validations, error) {
	result := Validations{}
	for _, group := range groups {
		finalized, err := WrapGroup(context, group)
		if err != nil {
			return Validations{}, err
		}
		result.Add(finalized)
	}
	return result, nil
}

// TypeValidator generates validations derived from collected metadata
// rather than from a single tag occurrence (e.g. unions assembled from many
// member tags). The registry invokes registered emitters for type-ish scopes
// and finalizes the returned groups.
type TypeValidator interface {
	Name() string
	ValidateType(context Context, metadata SchemaMetadata) ([]ValidationGroup, error)
}

// Aggregate emitter chain positions. Emission order determines the order of
// generated validations, which is visible in generated output.
const (
	TypeUnionsOrder = 100
	TypeModesOrder  = 200
)

type registeredTypeValidator struct {
	order   int
	emitter TypeValidator
}

var typeValidators []registeredTypeValidator

// RegisterTypeValidator adds an aggregate emitter at the given position.
// Positions must be unique so the emission order is unambiguous.
func RegisterTypeValidator(order int, e TypeValidator) {
	for _, existing := range typeValidators {
		if existing.order == order {
			panic(fmt.Sprintf("aggregate emitter order %d already taken by %q, cannot register %q", order, existing.emitter.Name(), e.Name()))
		}
	}
	typeValidators = append(typeValidators, registeredTypeValidator{order: order, emitter: e})
	slices.SortFunc(typeValidators, func(a, b registeredTypeValidator) int {
		return a.order - b.order
	})
}

// GenerateTypeValidations runs all registered aggregate emitters and
// finalizes their groups.
func GenerateTypeValidations(context Context, metadata SchemaMetadata) (Validations, error) {
	result := Validations{}
	for _, ra := range typeValidators {
		groups, err := ra.emitter.ValidateType(context, metadata)
		if err != nil {
			return Validations{}, fmt.Errorf("aggregate emitter %q: %w", ra.emitter.Name(), err)
		}
		finalized, err := WrapGroups(context, groups)
		if err != nil {
			return Validations{}, err
		}
		result.Add(finalized)
	}
	return result, nil
}
