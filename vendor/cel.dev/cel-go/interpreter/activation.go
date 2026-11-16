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

	"cel.dev/cel-go/common/functions"
	"cel.dev/cel-go/common/types/ref"
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

func (emptyActivation) Parent() Activation { return nil }

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
// the types.Adapter configured in the environment.
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

// mapActivation is the default Activation implementation, which supplies variables by name and,
// where the caller provided them, the implementations of late-bound functions.
//
// Variables and functions occupy separate namespaces, so the same qualified name may refer to
// both. An mapActivation may also delegate to the mapActivation whose bindings it extends, which is
// exposed as its parent so that the hierarchy is visible to all mapActivation traversals.
//
// Named bindings may lazily supply values by providing a function which accepts no arguments and
// produces an interface value.
type mapActivation struct {
	bindings map[string]any
	// TODO: validate that the late-bound functions should be applied here if at all.
	// Seems like it would be better if populated via setter?
	functions map[string]functions.LateBoundOp
	parent    Activation
}

// Parent implements the Activation interface method.
func (a *mapActivation) Parent() Activation {
	return a.parent
}

// ResolveName implements the Activation interface method.
func (a *mapActivation) ResolveName(name string) (any, bool) {
	obj, found := a.bindings[name]
	if !found {
		if a.parent != nil {
			return a.parent.ResolveName(name)
		}
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

// ResolveFunction implements the FunctionActivation interface method.
//
// Only the bindings supplied to this activation are considered. A hierarchy may contain at most
// one activation which supplies late-bound functions, so there is no parent to delegate to.
func (a *mapActivation) ResolveFunction(name string) (functions.LateBoundOp, bool) {
	fn, found := a.functions[name]
	return fn, found
}

// AsPartialActivation supports partial evaluation over the activation being extended.
func (a *mapActivation) AsPartialActivation() (PartialActivation, bool) {
	return AsPartialActivation(a.parent)
}

// hierarchicalActivation which implements Activation and contains a parent and
// child activation.
type hierarchicalActivation struct {
	parent        Activation
	child         Activation
	poolAllocated bool
}

// Parent implements the Activation interface method.
func (a *hierarchicalActivation) Parent() Activation {
	return a.parent
}

// ResolveName implements the Activation interface method.
func (a *hierarchicalActivation) ResolveName(name string) (any, bool) {
	if a.child != nil {
		if object, found := a.child.ResolveName(name); found {
			return object, found
		}
	}
	if a.parent != nil {
		return a.parent.ResolveName(name)
	}
	return nil, false
}

// Unwrap returns the parent activation, stripping the local child scope.
// This allows global disambiguation to skip past locally introduced variables.
func (a *hierarchicalActivation) Unwrap() Activation {
	return a.parent
}

// IsLocalVariable reports whether the variable name is locally bound in the hierarchical activation.
func (a *hierarchicalActivation) IsLocalVariable(name string) bool {
	if holder, ok := a.child.(localVariableHolder); ok {
		if holder.IsLocalVariable(name) {
			return true
		}
	}
	if holder, ok := a.parent.(localVariableHolder); ok {
		return holder.IsLocalVariable(name)
	}
	return false
}

// AsPartialActivation checks the child first via direct type assertion (to
// avoid recursion through the folder → frame → hierarchicalActivation cycle),
// then walks the parent hierarchy via the free function.
func (a *hierarchicalActivation) AsPartialActivation() (PartialActivation, bool) {
	if pv, ok := a.child.(partialActivationConverter); ok {
		if p, ok := pv.AsPartialActivation(); ok {
			return p, true
		}
	}
	return AsPartialActivation(a.parent)
}

// NewHierarchicalActivation takes two activations and produces a new one which prioritizes
// resolution in the child first and parent(s) second.
func NewHierarchicalActivation(parent Activation, child Activation) Activation {
	return &hierarchicalActivation{parent: parent, child: child, poolAllocated: false}
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

// partialActivationConverter indicates whether an Activation implementation supports conversion to a PartialActivation
type partialActivationConverter interface {
	// AsPartialActivation converts the current activation to a PartialActivation
	AsPartialActivation() (PartialActivation, bool)
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

// AsPartialActivation returns the partActivation as a PartialActivation interface.
func (a *partActivation) AsPartialActivation() (PartialActivation, bool) {
	return a, true
}

// AsPartialActivation walks the activation hierarchy and returns the first PartialActivation, if found.
func AsPartialActivation(vars Activation) (PartialActivation, bool) {
	if vars == nil {
		return nil, false
	}
	// Only internal activation instances may implement this interface
	if pv, ok := vars.(partialActivationConverter); ok {
		return pv.AsPartialActivation()
	}
	// Since Activations may be hierarchical, test whether a parent converts to a PartialActivation
	if vars.Parent() != nil {
		return AsPartialActivation(vars.Parent())
	}
	return nil, false
}

// FunctionActivation extends the Activation interface with late-bound function implementations.
//
// Function bindings occupy a namespace which is separate from variable bindings, so a function
// and a variable may share the same qualified name without ambiguity.
//
// The implementation which applies to an evaluation is located once, before evaluation begins, so
// at most one activation in a hierarchy may implement this interface. An implementation supplied
// by a caller is treated as the sole source of bindings and its parents are not searched.
type FunctionActivation interface {
	Activation

	// ResolveFunction returns the implementation of the late-bound function by qualified name,
	// or false if the function could not be found.
	ResolveFunction(name string) (functions.LateBoundOp, bool)
}

// NewFunctionActivation returns an Activation which supplies implementations for late-bound
// functions in addition to the variables provided by the `vars` input.
//
// The `vars` value may either be an Activation or any valid input to the NewActivation call, and
// the result is the same kind of activation that NewActivation produces, extended with the
// function bindings.
//
// Each function implementation is invoked with the id of the declared overload which matched the
// call arguments, which permits a single implementation to serve all overloads of the function.
//
// Late-bound functions are bound for the whole of an evaluation rather than for a lexical scope,
// so an error is returned when `vars` already supplies function bindings. Supply the complete set
// of bindings in a single call instead of layering them.
func NewFunctionActivation(vars any, funcs map[string]functions.LateBoundOp) (FunctionActivation, error) {
	// Copy the input to ensure the bindings observed during an evaluation cannot be mutated by
	// the caller once the activation has been created.
	bindings := make(map[string]functions.LateBoundOp, len(funcs))
	for name, fn := range funcs {
		if fn == nil {
			return nil, fmt.Errorf("function binding must be non-nil: %s", name)
		}
		bindings[name] = fn
	}
	if m, isMap := vars.(map[string]any); isMap {
		return &mapActivation{bindings: m, functions: bindings}, nil
	}
	a, err := NewActivation(vars)
	if err != nil {
		return nil, err
	}
	if err := checkNoFunctionActivation(a); err != nil {
		return nil, err
	}
	return &mapActivation{functions: bindings, parent: a}, nil
}

// errNestedFunctionActivation reports a hierarchy which supplies late-bound functions from more
// than one activation, where the bindings which apply would depend on the order in which the
// activations were composed.
var errNestedFunctionActivation = errors.New(
	"activation hierarchy may supply late-bound functions from only one activation")

// checkNoFunctionActivation returns an error if the hierarchy already supplies late-bound
// function implementations.
func checkNoFunctionActivation(vars Activation) error {
	found, err := FindFunctionActivation(vars)
	if err != nil {
		return err
	}
	if found != nil {
		return errNestedFunctionActivation
	}
	return nil
}

// FindFunctionActivation returns the activation which supplies the late-bound function
// implementations for a hierarchy, or nil when the hierarchy supplies none.
//
// Late-bound functions are bound for the duration of an evaluation rather than for a lexical
// scope, so a hierarchy may contain at most one activation which supplies them and an error is
// returned when it contains more. Resolving the binding once, before evaluation begins, keeps the
// lookup at a call site independent of how many scopes enclose it.
//
// This must be called before evaluation begins, while the hierarchy is still composed only of
// caller-supplied activations. Scopes introduced during evaluation, such as those of a
// comprehension, report the enclosing frame as their parent rather than the enclosing activation.
func FindFunctionActivation(vars Activation) (FunctionActivation, error) {
	return findFunctionActivation(vars, nil)
}

// findFunctionActivation searches a hierarchy for the activation which supplies late-bound
// functions, carrying the match found so far so that a second match can be reported as an error.
func findFunctionActivation(vars Activation, found FunctionActivation) (FunctionActivation, error) {
	if vars == nil {
		return found, nil
	}
	switch a := vars.(type) {
	case *mapActivation:
		// A default activation only supplies functions when it was created for that purpose.
		if a.functions != nil {
			if found != nil {
				return nil, errNestedFunctionActivation
			}
			found = a
		}
		return findFunctionActivation(a.parent, found)
	case *hierarchicalActivation:
		// Both scopes are caller-supplied, so both must be searched.
		found, err := findFunctionActivation(a.parent, found)
		if err != nil {
			return nil, err
		}
		return findFunctionActivation(a.child, found)
	case *partActivation:
		// The wrapped activation is not reachable via Parent.
		return findFunctionActivation(a.Activation, found)
	default:
		// An implementation supplied by the caller is responsible for the scopes it contains,
		// so its parents are not searched.
		if fa, isFunc := vars.(FunctionActivation); isFunc {
			if found != nil {
				return nil, errNestedFunctionActivation
			}
			return fa, nil
		}
		return findFunctionActivation(vars.Parent(), found)
	}
}
