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

package cel

import (
	"fmt"
	"sort"

	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// StaticOptimizer contains a sequence of ASTOptimizer instances which will be applied in order.
//
// The static optimizer normalizes expression ids and type-checking run between optimization
// passes to ensure that the final optimized output is a valid expression with metadata consistent
// with what would have been generated from a parsed and checked expression.
//
// Note: source position information is best-effort and incomplete, but optimized expressions
// should be suitable for calls to parser.Unparse.
type StaticOptimizer struct {
	optimizers []ASTOptimizer
	// If set, Optimize() will use this Source instead of the one from the AST.
	sourceOverride *Source
}

type OptimizerOption func(*StaticOptimizer) (*StaticOptimizer, error)

// NewStaticOptimizer creates a StaticOptimizer with a sequence of ASTOptimizer's to be applied
// to a checked expression.
func NewStaticOptimizer(options ...any) (*StaticOptimizer, error) {
	so := &StaticOptimizer{}
	var err error
	for _, opt := range options {
		switch v := opt.(type) {
		case ASTOptimizer:
			so.optimizers = append(so.optimizers, v)
		case OptimizerOption:
			so, err = v(so)
			if err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("unsupported option: %v", v)
		}
	}
	return so, nil
}

// OptimizeWithSource overrides the source used by the optimizer.
// Note this will cause the source info from the AST passed to Optimize() to be discarded.
func OptimizeWithSource(source Source) OptimizerOption {
	return func(so *StaticOptimizer) (*StaticOptimizer, error) {
		so.sourceOverride = &source
		return so, nil
	}
}

// Optimize applies a sequence of optimizations to an Ast within a given environment.
//
// If issues are encountered, the Issues.Err() return value will be non-nil.
func (opt *StaticOptimizer) Optimize(env *Env, a *Ast) (*Ast, *Issues) {
	// Make a copy of the AST to be optimized.
	optimized := ast.Copy(a.NativeRep())
	source := a.Source()
	sourceInfo := optimized.SourceInfo()
	if opt.sourceOverride != nil {
		source = *opt.sourceOverride
		sourceInfo = ast.NewSourceInfo(*opt.sourceOverride)
	}
	ids := newIDGenerator(ast.MaxID(a.NativeRep()))

	// Create the optimizer context, could be pooled in the future.
	issues := NewIssues(common.NewErrors(source))
	baseFac := ast.NewExprFactory()
	exprFac := &optimizerExprFactory{
		idGenerator: ids,
		fac:         baseFac,
		sourceInfo:  sourceInfo,
	}
	ctx := &OptimizerContext{
		optimizerExprFactory: exprFac,
		Env:                  env,
		Issues:               issues,
	}

	// Apply the optimizations sequentially.
	for _, o := range opt.optimizers {
		optimized = o.Optimize(ctx, optimized)
		if issues.Err() != nil {
			return nil, issues
		}
		// Normalize expression id metadata including coordination with macro call metadata.
		freshIDGen := newIDGenerator(0)
		info := optimized.SourceInfo()
		expr := optimized.Expr()
		normalizeIDs(freshIDGen.renumberStable, expr, info)
		cleanupMacroRefs(expr, info)

		// Recheck the updated expression for any possible type-agreement or validation errors.
		parsed := &Ast{
			source: source,
			impl:   ast.NewAST(expr, info)}
		checked, iss := ctx.Check(parsed)
		if iss.Err() != nil {
			return nil, iss
		}
		optimized = checked.NativeRep()
	}

	// Return the optimized result.
	return &Ast{
		source: source,
		impl:   optimized,
	}, nil
}

// normalizeIDs ensures that the metadata present with an AST is reset in a manner such
// that the ids within the expression correspond to the ids within macros.
func normalizeIDs(idGen ast.IDGenerator, optimized ast.Expr, info *ast.SourceInfo) {
	optimized.RenumberIDs(idGen)
	info.RenumberIDs(idGen)

	if len(info.MacroCalls()) == 0 {
		return
	}

	// Sort the macro ids to make sure that the renumbering of macro-specific variables
	// is stable across normalization calls.
	sortedMacroIDs := []int64{}
	for id := range info.MacroCalls() {
		sortedMacroIDs = append(sortedMacroIDs, id)
	}
	sort.Slice(sortedMacroIDs, func(i, j int) bool { return sortedMacroIDs[i] < sortedMacroIDs[j] })

	// First, update the macro call ids themselves.
	callIDMap := map[int64]int64{}
	for _, id := range sortedMacroIDs {
		callIDMap[id] = idGen(id)
	}
	// Then update the macro call definitions which refer to these ids, but
	// ensure that the updates don't collide and remove macro entries which haven't
	// been visited / updated yet.
	type macroUpdate struct {
		id   int64
		call ast.Expr
	}
	macroUpdates := []macroUpdate{}
	for _, oldID := range sortedMacroIDs {
		newID := callIDMap[oldID]
		call, found := info.GetMacroCall(oldID)
		if !found {
			continue
		}
		call.RenumberIDs(idGen)
		macroUpdates = append(macroUpdates, macroUpdate{id: newID, call: call})
		info.ClearMacroCall(oldID)
	}
	for _, u := range macroUpdates {
		info.SetMacroCall(u.id, u.call)
	}
}

func cleanupMacroRefs(expr ast.Expr, info *ast.SourceInfo) {
	if len(info.MacroCalls()) == 0 {
		return
	}

	// Sanitize the macro call references once the optimized expression has been computed
	// and the ids normalized between the expression and the macros.
	exprRefMap := make(map[int64]struct{})
	ast.PostOrderVisit(expr, ast.NewExprVisitor(func(e ast.Expr) {
		if e.ID() == 0 {
			return
		}
		exprRefMap[e.ID()] = struct{}{}
	}))
	// Update the macro call id references to ensure that macro pointers are
	// updated consistently across macros.
	for _, call := range info.MacroCalls() {
		ast.PostOrderVisit(call, ast.NewExprVisitor(func(e ast.Expr) {
			if e.ID() == 0 {
				return
			}
			exprRefMap[e.ID()] = struct{}{}
		}))
	}
	for id := range info.MacroCalls() {
		if _, found := exprRefMap[id]; !found {
			info.ClearMacroCall(id)
		}
	}
}

// newIDGenerator ensures that new ids are only created the first time they are encountered.
func newIDGenerator(seed int64) *idGenerator {
	return &idGenerator{
		idMap: make(map[int64]int64),
		seed:  seed,
	}
}

type idGenerator struct {
	idMap map[int64]int64
	seed  int64
}

func (gen *idGenerator) nextID() int64 {
	gen.seed++
	return gen.seed
}

func (gen *idGenerator) renumberStable(id int64) int64 {
	if id == 0 {
		return 0
	}
	if newID, found := gen.idMap[id]; found {
		return newID
	}
	nextID := gen.nextID()
	gen.idMap[id] = nextID
	return nextID
}

// OptimizerContext embeds Env and Issues instances to make it easy to type-check and evaluate
// subexpressions and report any errors encountered along the way. The context also embeds the
// optimizerExprFactory which can be used to generate new sub-expressions with expression ids
// consistent with the expectations of a parsed expression.
type OptimizerContext struct {
	*Env
	*optimizerExprFactory
	*Issues
}

// ExtendEnv auguments the context's environment with the additional options.
func (opt *OptimizerContext) ExtendEnv(opts ...EnvOption) error {
	e, err := opt.Env.Extend(opts...)
	if err != nil {
		return err
	}
	opt.Env = e
	return nil
}

// ASTOptimizer applies an optimization over an AST and returns the optimized result.
type ASTOptimizer interface {
	// Optimize optimizes a type-checked AST within an Environment and accumulates any issues.
	Optimize(*OptimizerContext, *ast.AST) *ast.AST
}

type optimizerExprFactory struct {
	*idGenerator
	fac        ast.ExprFactory
	sourceInfo *ast.SourceInfo
}

// NewAST creates an AST from the current expression using the tracked source info which
// is modified and managed by the OptimizerContext.
func (opt *optimizerExprFactory) NewAST(expr ast.Expr) *ast.AST {
	return ast.NewAST(expr, opt.sourceInfo)
}

// CopyAST creates a renumbered copy of `Expr` and `SourceInfo` values of the input AST, where the
// renumbering uses the same scheme as the core optimizer logic ensuring there are no collisions
// between copies.
//
// Use this method before attempting to merge the expression from AST into another.
func (opt *optimizerExprFactory) CopyAST(a *ast.AST) (ast.Expr, *ast.SourceInfo) {
	idGen := newIDGenerator(opt.nextID())
	defer func() { opt.seed = idGen.nextID() }()
	copyExpr := opt.fac.CopyExpr(a.Expr())
	copyInfo := ast.CopySourceInfo(a.SourceInfo())
	normalizeIDs(idGen.renumberStable, copyExpr, copyInfo)
	return copyExpr, copyInfo
}

// CopyASTAndMetadata copies the input AST and propagates the macro metadata into the AST being
// optimized.
func (opt *optimizerExprFactory) CopyASTAndMetadata(a *ast.AST) ast.Expr {
	copyExpr, copyInfo := opt.CopyAST(a)
	for macroID, call := range copyInfo.MacroCalls() {
		opt.SetMacroCall(macroID, call)
	}
	for id, offset := range copyInfo.OffsetRanges() {
		opt.sourceInfo.SetOffsetRange(id, offset)
	}
	return copyExpr
}

// ClearMacroCall clears the macro at the given expression id.
func (opt *optimizerExprFactory) ClearMacroCall(id int64) {
	opt.sourceInfo.ClearMacroCall(id)
}

// SetMacroCall sets the macro call metadata for the given macro id within the tracked source info
// metadata.
func (opt *optimizerExprFactory) SetMacroCall(id int64, expr ast.Expr) {
	opt.sourceInfo.SetMacroCall(id, expr)
}

// MacroCalls returns the map of macro calls currently in the context.
func (opt *optimizerExprFactory) MacroCalls() map[int64]ast.Expr {
	return opt.sourceInfo.MacroCalls()
}

// NewBindMacro creates an AST expression representing the expanded bind() macro, and a macro expression
// representing the unexpanded call signature to be inserted into the source info macro call metadata.
func (opt *optimizerExprFactory) NewBindMacro(macroID int64, varName string, varInit, remaining ast.Expr) (astExpr, macroExpr ast.Expr) {
	varID := opt.nextID()
	remainingID := opt.nextID()
	remaining = opt.fac.CopyExpr(remaining)
	remaining.RenumberIDs(func(id int64) int64 {
		if id == macroID {
			return remainingID
		}
		return id
	})
	if call, exists := opt.sourceInfo.GetMacroCall(macroID); exists {
		opt.SetMacroCall(remainingID, opt.fac.CopyExpr(call))
	}

	astExpr = opt.fac.NewComprehension(macroID,
		opt.fac.NewList(opt.nextID(), []ast.Expr{}, []int32{}),
		"#unused",
		varName,
		opt.fac.CopyExpr(varInit),
		opt.fac.NewLiteral(opt.nextID(), types.False),
		opt.fac.NewIdent(varID, varName),
		remaining)

	macroExpr = opt.fac.NewMemberCall(0, "bind",
		opt.fac.NewIdent(opt.nextID(), "cel"),
		opt.fac.NewIdent(varID, varName),
		opt.fac.CopyExpr(varInit),
		opt.fac.CopyExpr(remaining))
	opt.sanitizeMacro(macroID, macroExpr)
	return
}

// NewCall creates a global function call invocation expression.
//
// Example:
//
// countByField(list, fieldName)
// - function: countByField
// - args: [list, fieldName]
func (opt *optimizerExprFactory) NewCall(function string, args ...ast.Expr) ast.Expr {
	return opt.fac.NewCall(opt.nextID(), function, args...)
}

// NewMemberCall creates a member function call invocation expression where 'target' is the receiver of the call.
//
// Example:
//
// list.countByField(fieldName)
// - function: countByField
// - target: list
// - args: [fieldName]
func (opt *optimizerExprFactory) NewMemberCall(function string, target ast.Expr, args ...ast.Expr) ast.Expr {
	return opt.fac.NewMemberCall(opt.nextID(), function, target, args...)
}

// NewIdent creates a new identifier expression.
//
// Examples:
//
// - simple_var_name
// - qualified.subpackage.var_name
func (opt *optimizerExprFactory) NewIdent(name string) ast.Expr {
	return opt.fac.NewIdent(opt.nextID(), name)
}

// NewLiteral creates a new literal expression value.
//
// The range of valid values for a literal generated during optimization is different than for expressions
// generated via parsing / type-checking, as the ref.Val may be _any_ CEL value so long as the value can
// be converted back to a literal-like form.
func (opt *optimizerExprFactory) NewLiteral(value ref.Val) ast.Expr {
	return opt.fac.NewLiteral(opt.nextID(), value)
}

// NewList creates a list expression with a set of optional indices.
//
// Examples:
//
// [a, b]
// - elems: [a, b]
// - optIndices: []
//
// [a, ?b, ?c]
// - elems: [a, b, c]
// - optIndices: [1, 2]
func (opt *optimizerExprFactory) NewList(elems []ast.Expr, optIndices []int32) ast.Expr {
	return opt.fac.NewList(opt.nextID(), elems, optIndices)
}

// NewMap creates a map from a set of entry expressions which contain a key and value expression.
func (opt *optimizerExprFactory) NewMap(entries []ast.EntryExpr) ast.Expr {
	return opt.fac.NewMap(opt.nextID(), entries)
}

// NewMapEntry creates a map entry with a key and value expression and a flag to indicate whether the
// entry is optional.
//
// Examples:
//
// {a: b}
// - key: a
// - value: b
// - optional: false
//
// {?a: ?b}
// - key: a
// - value: b
// - optional: true
func (opt *optimizerExprFactory) NewMapEntry(key, value ast.Expr, isOptional bool) ast.EntryExpr {
	return opt.fac.NewMapEntry(opt.nextID(), key, value, isOptional)
}

// NewHasMacro generates a test-only select expression to be included within an AST and an unexpanded
// has() macro call signature to be inserted into the source info macro call metadata.
func (opt *optimizerExprFactory) NewHasMacro(macroID int64, s ast.Expr) (astExpr, macroExpr ast.Expr) {
	sel := s.AsSelect()
	astExpr = opt.fac.NewPresenceTest(macroID, sel.Operand(), sel.FieldName())
	macroExpr = opt.fac.NewCall(0, "has",
		opt.NewSelect(opt.fac.CopyExpr(sel.Operand()), sel.FieldName()))
	opt.sanitizeMacro(macroID, macroExpr)
	return
}

// NewSelect creates a select expression where a field value is selected from an operand.
//
// Example:
//
// msg.field_name
// - operand: msg
// - field: field_name
func (opt *optimizerExprFactory) NewSelect(operand ast.Expr, field string) ast.Expr {
	return opt.fac.NewSelect(opt.nextID(), operand, field)
}

// NewStruct creates a new typed struct value with an set of field initializations.
//
// Example:
//
// pkg.TypeName{field: value}
// - typeName: pkg.TypeName
// - fields: [{field: value}]
func (opt *optimizerExprFactory) NewStruct(typeName string, fields []ast.EntryExpr) ast.Expr {
	return opt.fac.NewStruct(opt.nextID(), typeName, fields)
}

// NewStructField creates a struct field initialization.
//
// Examples:
//
// {count: 3u}
// - field: count
// - value: 3u
// - optional: false
//
// {?count: x}
// - field: count
// - value: x
// - optional: true
func (opt *optimizerExprFactory) NewStructField(field string, value ast.Expr, isOptional bool) ast.EntryExpr {
	return opt.fac.NewStructField(opt.nextID(), field, value, isOptional)
}

// UpdateExpr updates the target expression with the updated content while preserving macro metadata.
//
// There are four scenarios during the update to consider:
// 1. target is not macro, updated is not macro
// 2. target is macro, updated is not macro
// 3. target is macro, updated is macro
// 4. target is not macro, updated is macro
//
// When the target is a macro already, it may either be updated to a new macro function
// body if the update is also a macro, or it may be removed altogether if the update is
// a macro.
//
// When the update is a macro, then the target references within other macros must be
// updated to point to the new updated macro. Otherwise, other macros which pointed to
// the target body must be replaced with copies of the updated expression body.
func (opt *optimizerExprFactory) UpdateExpr(target, updated ast.Expr) {
	// Update the expression
	target.SetKindCase(updated)

	// Early return if there's no macros present sa the source info reflects the
	// macro set from the target and updated expressions.
	if len(opt.sourceInfo.MacroCalls()) == 0 {
		return
	}
	// Determine whether the target expression was a macro.
	_, targetIsMacro := opt.sourceInfo.GetMacroCall(target.ID())

	// Determine whether the updated expression was a macro.
	updatedMacro, updatedIsMacro := opt.sourceInfo.GetMacroCall(updated.ID())

	if updatedIsMacro {
		// If the updated call was a macro, then updated id maps to target id,
		// and the updated macro moves into the target id slot.
		opt.sourceInfo.ClearMacroCall(updated.ID())
		opt.sourceInfo.SetMacroCall(target.ID(), updatedMacro)
	} else if targetIsMacro {
		// Otherwise if the target expr was a macro, but is no longer, clear
		// the macro reference.
		opt.sourceInfo.ClearMacroCall(target.ID())
	}

	// Punch holes in the updated value where macros references exist.
	macroExpr := opt.fac.CopyExpr(target)
	macroRefVisitor := ast.NewExprVisitor(func(e ast.Expr) {
		if _, exists := opt.sourceInfo.GetMacroCall(e.ID()); exists {
			e.SetKindCase(nil)
		}
	})
	ast.PostOrderVisit(macroExpr, macroRefVisitor)

	// Update any references to the expression within a macro
	macroVisitor := ast.NewExprVisitor(func(call ast.Expr) {
		// Update the target expression to point to the macro expression which
		// will be empty if the updated expression was a macro.
		if call.ID() == target.ID() {
			call.SetKindCase(opt.fac.CopyExpr(macroExpr))
		}
		// Update the macro call expression if it refers to the updated expression
		// id which has since been remapped to the target id.
		if call.ID() == updated.ID() {
			// Either ensure the expression is a macro reference or a populated with
			// the relevant sub-expression if the updated expr was not a macro.
			if updatedIsMacro {
				call.SetKindCase(nil)
			} else {
				call.SetKindCase(opt.fac.CopyExpr(macroExpr))
			}
			// Since SetKindCase does not renumber the id, ensure the references to
			// the old 'updated' id are mapped to the target id.
			call.RenumberIDs(func(id int64) int64 {
				if id == updated.ID() {
					return target.ID()
				}
				return id
			})
		}
	})
	for _, call := range opt.sourceInfo.MacroCalls() {
		ast.PostOrderVisit(call, macroVisitor)
	}
}

func (opt *optimizerExprFactory) sanitizeMacro(macroID int64, macroExpr ast.Expr) {
	macroRefVisitor := ast.NewExprVisitor(func(e ast.Expr) {
		if _, exists := opt.sourceInfo.GetMacroCall(e.ID()); exists && e.ID() != macroID {
			e.SetKindCase(nil)
		}
	})
	ast.PostOrderVisit(macroExpr, macroRefVisitor)
}
