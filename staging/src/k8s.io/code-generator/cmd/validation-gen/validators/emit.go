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

// EmittedGroup is a batch of validations plus declarative facts about where
// and when they apply. Producers (e.g. the union or item generators) state
// facts; they never apply wrapping themselves. Each fact is consumed by the
// EmissionFinalizer registered by the validator that owns that concern:
// Conditions by options.go, StabilityLevel by levels.go, TargetPath by
// subfield.go.
type EmittedGroup struct {
	Validations Validations

	// Conditions that must hold for these validations to run.
	Conditions Conditions

	// StabilityLevel to mark these validations with, if any.
	StabilityLevel ValidationStabilityLevel

	// TargetPath is the absolute schema path the validations belong to. When
	// it names a descendant of the emitting context's path, the validations
	// are re-homed there. Nil means the current context.
	TargetPath *field.Path

	// TargetType is the type at TargetPath, used as the object type for
	// generated wrapper functions. Nil means the current context's type.
	TargetType *types.Type

	// Hoist indicates the finalized functions belong on the enclosing type's
	// validation function: after the finalizer chain runs, they are moved to
	// TypeFunctions. Variables and comments are unaffected.
	Hoist bool
}

// EmissionFinalizer materializes one cross-cutting concern of an
// EmittedGroup. A finalizer consumes its envelope field(s), clearing them
// when applied, and must be a no-op passthrough when its field is empty.
type EmissionFinalizer interface {
	Name() string
	Finalize(context Context, group EmittedGroup) (EmittedGroup, error)
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
	finalizer EmissionFinalizer
}

var emissionFinalizers []registeredFinalizer

// RegisterEmissionFinalizer adds a finalizer to the emission chain at the
// given position. Positions must be unique so the chain order is unambiguous.
func RegisterEmissionFinalizer(order int, f EmissionFinalizer) {
	for _, existing := range emissionFinalizers {
		if existing.order == order {
			panic(fmt.Sprintf("emission finalizer order %d already taken by %q, cannot register %q", order, existing.finalizer.Name(), f.Name()))
		}
	}
	emissionFinalizers = append(emissionFinalizers, registeredFinalizer{order: order, finalizer: f})
	slices.SortFunc(emissionFinalizers, func(a, b registeredFinalizer) int {
		return a.order - b.order
	})
}

// FinalizeGroup runs the emission finalizer chain over one group and returns
// the resulting validations.
func FinalizeGroup(context Context, group EmittedGroup) (Validations, error) {
	for _, rf := range emissionFinalizers {
		var err error
		group, err = rf.finalizer.Finalize(context, group)
		if err != nil {
			return Validations{}, fmt.Errorf("emission finalizer %q: %w", rf.finalizer.Name(), err)
		}
	}
	if group.Hoist {
		group.Validations.TypeFunctions = append(group.Validations.TypeFunctions, group.Validations.Functions...)
		group.Validations.Functions = nil
	}
	return group.Validations, nil
}

// FinalizeGroups runs the emission finalizer chain over each group and merges
// the results.
func FinalizeGroups(context Context, groups []EmittedGroup) (Validations, error) {
	result := Validations{}
	for _, group := range groups {
		finalized, err := FinalizeGroup(context, group)
		if err != nil {
			return Validations{}, err
		}
		result.Add(finalized)
	}
	return result, nil
}

// AggregateEmitter generates validations derived from collected metadata
// rather than from a single tag occurrence (e.g. unions assembled from many
// member tags). The registry invokes registered emitters for type-ish scopes
// and finalizes the returned groups.
type AggregateEmitter interface {
	Name() string
	GenerateGroups(context Context, metadata SchemaMetadata) ([]EmittedGroup, error)
}

// Aggregate emitter chain positions. Emission order determines the order of
// generated validations, which is visible in generated output.
const (
	AggregateUnionsOrder = 100
	AggregateModesOrder  = 200
)

type registeredAggregate struct {
	order   int
	emitter AggregateEmitter
}

var aggregateEmitters []registeredAggregate

// RegisterAggregateEmitter adds an aggregate emitter at the given position.
// Positions must be unique so the emission order is unambiguous.
func RegisterAggregateEmitter(order int, e AggregateEmitter) {
	for _, existing := range aggregateEmitters {
		if existing.order == order {
			panic(fmt.Sprintf("aggregate emitter order %d already taken by %q, cannot register %q", order, existing.emitter.Name(), e.Name()))
		}
	}
	aggregateEmitters = append(aggregateEmitters, registeredAggregate{order: order, emitter: e})
	slices.SortFunc(aggregateEmitters, func(a, b registeredAggregate) int {
		return a.order - b.order
	})
}

// GenerateAggregateValidations runs all registered aggregate emitters and
// finalizes their groups.
func GenerateAggregateValidations(context Context, metadata SchemaMetadata) (Validations, error) {
	result := Validations{}
	for _, ra := range aggregateEmitters {
		groups, err := ra.emitter.GenerateGroups(context, metadata)
		if err != nil {
			return Validations{}, fmt.Errorf("aggregate emitter %q: %w", ra.emitter.Name(), err)
		}
		finalized, err := FinalizeGroups(context, groups)
		if err != nil {
			return Validations{}, err
		}
		result.Add(finalized)
	}
	return result, nil
}
