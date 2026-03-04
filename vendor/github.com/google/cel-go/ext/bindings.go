// Copyright 2023 Google LLC
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

package ext

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/interpreter"
)

// Bindings returns a cel.EnvOption to configure support for local variable
// bindings in expressions.
//
// # Cel.Bind
//
// Binds a simple identifier to an initialization expression which may be used
// in a subsequenct result expression. Bindings may also be nested within each
// other.
//
//	cel.bind(<varName>, <initExpr>, <resultExpr>)
//
// Examples:
//
//	cel.bind(a, 'hello',
//	cel.bind(b, 'world', a + b + b + a)) // "helloworldworldhello"
//
//	// Avoid a list allocation within the exists comprehension.
//	cel.bind(valid_values, [a, b, c],
//	[d, e, f].exists(elem, elem in valid_values))
//
// Local bindings are not guaranteed to be evaluated before use.
func Bindings(options ...BindingsOption) cel.EnvOption {
	b := &celBindings{version: math.MaxUint32}
	for _, o := range options {
		b = o(b)
	}
	return cel.Lib(b)
}

const (
	celNamespace  = "cel"
	bindMacro     = "bind"
	blockFunc     = "@block"
	unusedIterVar = "#unused"
)

// BindingsOption declares a functional operator for configuring the Bindings library behavior.
type BindingsOption func(*celBindings) *celBindings

// BindingsVersion sets the version of the bindings library to an explicit version.
func BindingsVersion(version uint32) BindingsOption {
	return func(lib *celBindings) *celBindings {
		lib.version = version
		return lib
	}
}

type celBindings struct {
	version uint32
}

func (*celBindings) LibraryName() string {
	return "cel.lib.ext.cel.bindings"
}

func (lib *celBindings) CompileOptions() []cel.EnvOption {
	opts := []cel.EnvOption{
		cel.Macros(
			// cel.bind(var, <init>, <expr>)
			cel.ReceiverMacro(bindMacro, 3, celBind),
		),
	}
	if lib.version >= 1 {
		// The cel.@block signature takes a list of subexpressions and a typed expression which is
		// used as the output type.
		paramType := cel.TypeParamType("T")
		opts = append(opts,
			cel.Function("cel.@block",
				cel.Overload("cel_block_list",
					[]*cel.Type{cel.ListType(cel.DynType), paramType}, paramType)),
		)
		opts = append(opts, cel.ASTValidators(blockValidationExemption{}))
	}
	return opts
}

func (lib *celBindings) ProgramOptions() []cel.ProgramOption {
	if lib.version >= 1 {
		celBlockPlan := func(i interpreter.Interpretable) (interpreter.Interpretable, error) {
			call, ok := i.(interpreter.InterpretableCall)
			if !ok {
				return i, nil
			}
			switch call.Function() {
			case "cel.@block":
				args := call.Args()
				if len(args) != 2 {
					return nil, fmt.Errorf("cel.@block expects two arguments, but got %d", len(args))
				}
				expr := args[1]
				// Non-empty block
				if block, ok := args[0].(interpreter.InterpretableConstructor); ok {
					slotExprs := block.InitVals()
					return newDynamicBlock(slotExprs, expr), nil
				}
				// Constant valued block which can happen during runtime optimization.
				if cons, ok := args[0].(interpreter.InterpretableConst); ok {
					if cons.Value().Type() == types.ListType {
						l := cons.Value().(traits.Lister)
						if l.Size().Equal(types.IntZero) == types.True {
							return args[1], nil
						}
						return newConstantBlock(l, expr), nil
					}
				}
				return nil, errors.New("cel.@block expects a list constructor as the first argument")
			default:
				return i, nil
			}
		}
		return []cel.ProgramOption{cel.CustomDecorator(celBlockPlan)}
	}
	return []cel.ProgramOption{}
}

type blockValidationExemption struct{}

// Name returns the name of the validator.
func (blockValidationExemption) Name() string {
	return "cel.validator.cel_block"
}

// Configure implements the ASTValidatorConfigurer interface and augments the list of functions to skip
// during homogeneous aggregate literal type-checks.
func (blockValidationExemption) Configure(config cel.MutableValidatorConfig) error {
	functions := config.GetOrDefault(cel.HomogeneousAggregateLiteralExemptFunctions, []string{}).([]string)
	functions = append(functions, "cel.@block")
	return config.Set(cel.HomogeneousAggregateLiteralExemptFunctions, functions)
}

// Validate is a no-op as the intent is to simply disable strong type-checks for list literals during
// when they occur within cel.@block calls as the arg types have already been validated.
func (blockValidationExemption) Validate(env *cel.Env, _ cel.ValidatorConfig, a *ast.AST, iss *cel.Issues) {
}

func celBind(mef cel.MacroExprFactory, target ast.Expr, args []ast.Expr) (ast.Expr, *cel.Error) {
	if !macroTargetMatchesNamespace(celNamespace, target) {
		return nil, nil
	}
	varIdent := args[0]
	varName := ""
	switch varIdent.Kind() {
	case ast.IdentKind:
		varName = varIdent.AsIdent()
	default:
		return nil, mef.NewError(varIdent.ID(), "cel.bind() variable names must be simple identifiers")
	}
	varInit := args[1]
	resultExpr := args[2]
	return mef.NewComprehension(
		mef.NewList(),
		unusedIterVar,
		varName,
		varInit,
		mef.NewLiteral(types.False),
		mef.NewIdent(varName),
		resultExpr,
	), nil
}

func newDynamicBlock(slotExprs []interpreter.Interpretable, expr interpreter.Interpretable) interpreter.Interpretable {
	bs := &dynamicBlock{
		slotExprs: slotExprs,
		expr:      expr,
	}
	bs.slotActivationPool = &sync.Pool{
		New: func() any {
			slotCount := len(slotExprs)
			sa := &dynamicSlotActivation{
				slotExprs: slotExprs,
				slotCount: slotCount,
				slotVals:  make([]*slotVal, slotCount),
			}
			for i := 0; i < slotCount; i++ {
				sa.slotVals[i] = &slotVal{}
			}
			return sa
		},
	}
	return bs
}

type dynamicBlock struct {
	slotExprs          []interpreter.Interpretable
	expr               interpreter.Interpretable
	slotActivationPool *sync.Pool
}

// ID implements the Interpretable interface method.
func (b *dynamicBlock) ID() int64 {
	return b.expr.ID()
}

// Eval implements the Interpretable interface method.
func (b *dynamicBlock) Eval(activation cel.Activation) ref.Val {
	sa := b.slotActivationPool.Get().(*dynamicSlotActivation)
	sa.Activation = activation
	defer b.clearSlots(sa)
	return b.expr.Eval(sa)
}

func (b *dynamicBlock) clearSlots(sa *dynamicSlotActivation) {
	sa.reset()
	b.slotActivationPool.Put(sa)
}

type slotVal struct {
	value   *ref.Val
	visited bool
}

type dynamicSlotActivation struct {
	cel.Activation
	slotExprs []interpreter.Interpretable
	slotCount int
	slotVals  []*slotVal
}

// ResolveName implements the Activation interface method but handles variables prefixed with `@index`
// as special variables which exist within the slot-based memory of the cel.@block() where each slot
// refers to an expression which must be computed only once.
func (sa *dynamicSlotActivation) ResolveName(name string) (any, bool) {
	if idx, found := matchSlot(name, sa.slotCount); found {
		v := sa.slotVals[idx]
		if v.visited {
			// Return not found if the index expression refers to itself
			if v.value == nil {
				return nil, false
			}
			return *v.value, true
		}
		v.visited = true
		val := sa.slotExprs[idx].Eval(sa)
		v.value = &val
		return val, true
	}
	return sa.Activation.ResolveName(name)
}

func (sa *dynamicSlotActivation) reset() {
	sa.Activation = nil
	for _, sv := range sa.slotVals {
		sv.visited = false
		sv.value = nil
	}
}

func newConstantBlock(slots traits.Lister, expr interpreter.Interpretable) interpreter.Interpretable {
	count := slots.Size().(types.Int)
	return &constantBlock{slots: slots, slotCount: int(count), expr: expr}
}

type constantBlock struct {
	slots     traits.Lister
	slotCount int
	expr      interpreter.Interpretable
}

// ID implements the interpreter.Interpretable interface method.
func (b *constantBlock) ID() int64 {
	return b.expr.ID()
}

// Eval implements the interpreter.Interpretable interface method, and will proxy @index prefixed variable
// lookups into a set of constant slots determined from the plan step.
func (b *constantBlock) Eval(activation cel.Activation) ref.Val {
	vars := constantSlotActivation{Activation: activation, slots: b.slots, slotCount: b.slotCount}
	return b.expr.Eval(vars)
}

type constantSlotActivation struct {
	cel.Activation
	slots     traits.Lister
	slotCount int
}

// ResolveName implements Activation interface method and proxies @index prefixed lookups into the slot
// activation associated with the block scope.
func (sa constantSlotActivation) ResolveName(name string) (any, bool) {
	if idx, found := matchSlot(name, sa.slotCount); found {
		return sa.slots.Get(types.Int(idx)), true
	}
	return sa.Activation.ResolveName(name)
}

func matchSlot(name string, slotCount int) (int, bool) {
	if idx, found := strings.CutPrefix(name, indexPrefix); found {
		idx, err := strconv.Atoi(idx)
		// Return not found if the index is not numeric
		if err != nil {
			return -1, false
		}
		// Return not found if the index is not a valid slot
		if idx < 0 || idx >= slotCount {
			return -1, false
		}
		return idx, true
	}
	return -1, false
}

var (
	indexPrefix = "@index"
)
