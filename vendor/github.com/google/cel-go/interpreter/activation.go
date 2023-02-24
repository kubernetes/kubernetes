// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package interpreter

import (
	"errors"
	"fmt"
	"sync"

	"github.com/google/cel-go/common/types/ref"
)

// Activation used to resolve identifiers by name and references by id.
//
// An Activation is the primary mechanism by which a caller supplies input into a CEL program.
type Activation interface {
	// ResolveName returns a value from the activation by qualified name, or false if the name
	// could not be found.
	ResolveName(name string) (any, bool)

	// Parent returns the parent of the current activation, may be nil.
	// If non-nil, the parent will be searched during resolve calls.
	Parent() Activation
}

// EmptyActivation returns a variable-free activation.
func EmptyActivation() Activation {
	return emptyActivation{}
}

// emptyActivation is a variable-free activation.
type emptyActivation struct{}

func (emptyActivation) ResolveName(string) (any, bool) { return nil, false }
func (emptyActivation) Parent() Activation             { return nil }

// NewActivation returns an activation based on a map-based binding where the map keys are
// expected to be qualified names used with ResolveName calls.
//
// The input `bindings` may either be of type `Activation` or `map[string]any`.
//
// Lazy bindings may be supplied within the map-based input in either of the following forms:
// - func() any
// - func() ref.Val
//
// The output of the lazy binding will overwrite the variable reference in the internal map.
//
// Values which are not represented as ref.Val types on input may be adapted to a ref.Val using
// the ref.TypeAdapter configured in the environment.
func NewActivation(bindings any) (Activation, error) {
	if bindings == nil {
		return nil, errors.New("bindings must be non-nil")
	}
	a, isActivation := bindings.(Activation)
	if isActivation {
		return a, nil
	}
	m, isMap := bindings.(map[string]any)
	if !isMap {
		return nil, fmt.Errorf(
			"activation input must be an activation or map[string]interface: got %T",
			bindings)
	}
	return &mapActivation{bindings: m}, nil
}

// mapActivation which implements Activation and maps of named values.
//
// Named bindings may lazily supply values by providing a function which accepts no arguments and
// produces an interface value.
type mapActivation struct {
	bindings map[string]any
}

// Parent implements the Activation interface method.
func (a *mapActivation) Parent() Activation {
	return nil
}

// ResolveName implements the Activation interface method.
func (a *mapActivation) ResolveName(name string) (any, bool) {
	obj, found := a.bindings[name]
	if !found {
		return nil, false
	}
	fn, isLazy := obj.(func() ref.Val)
	if isLazy {
		obj = fn()
		a.bindings[name] = obj
	}
	fnRaw, isLazy := obj.(func() any)
	if isLazy {
		obj = fnRaw()
		a.bindings[name] = obj
	}
	return obj, found
}

// hierarchicalActivation which implements Activation and contains a parent and
// child activation.
type hierarchicalActivation struct {
	parent Activation
	child  Activation
}

// Parent implements the Activation interface method.
func (a *hierarchicalActivation) Parent() Activation {
	return a.parent
}

// ResolveName implements the Activation interface method.
func (a *hierarchicalActivation) ResolveName(name string) (any, bool) {
	if object, found := a.child.ResolveName(name); found {
		return object, found
	}
	return a.parent.ResolveName(name)
}

// NewHierarchicalActivation takes two activations and produces a new one which prioritizes
// resolution in the child first and parent(s) second.
func NewHierarchicalActivation(parent Activation, child Activation) Activation {
	return &hierarchicalActivation{parent, child}
}

// NewPartialActivation returns an Activation which contains a list of AttributePattern values
// representing field and index operations that should result in a 'types.Unknown' result.
//
// The `bindings` value may be any value type supported by the interpreter.NewActivation call,
// but is typically either an existing Activation or map[string]any.
func NewPartialActivation(bindings any,
	unknowns ...*AttributePattern) (PartialActivation, error) {
	a, err := NewActivation(bindings)
	if err != nil {
		return nil, err
	}
	return &partActivation{Activation: a, unknowns: unknowns}, nil
}

// PartialActivation extends the Activation interface with a set of UnknownAttributePatterns.
type PartialActivation interface {
	Activation

	// UnknownAttributePaths returns a set of AttributePattern values which match Attribute
	// expressions for data accesses whose values are not yet known.
	UnknownAttributePatterns() []*AttributePattern
}

// partActivation is the default implementations of the PartialActivation interface.
type partActivation struct {
	Activation
	unknowns []*AttributePattern
}

// UnknownAttributePatterns implements the PartialActivation interface method.
func (a *partActivation) UnknownAttributePatterns() []*AttributePattern {
	return a.unknowns
}

// varActivation represents a single mutable variable binding.
//
// This activation type should only be used within folds as the fold loop controls the object
// life-cycle.
type varActivation struct {
	parent Activation
	name   string
	val    ref.Val
}

// Parent implements the Activation interface method.
func (v *varActivation) Parent() Activation {
	return v.parent
}

// ResolveName implements the Activation interface method.
func (v *varActivation) ResolveName(name string) (any, bool) {
	if name == v.name {
		return v.val, true
	}
	return v.parent.ResolveName(name)
}

var (
	// pool of var activations to reduce allocations during folds.
	varActivationPool = &sync.Pool{
		New: func() any {
			return &varActivation{}
		},
	}
)
