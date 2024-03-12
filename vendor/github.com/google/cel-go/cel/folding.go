// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cel

import (
	"fmt"

	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

// ConstantFoldingOption defines a functional option for configuring constant folding.
type ConstantFoldingOption func(opt *constantFoldingOptimizer) (*constantFoldingOptimizer, error)

// MaxConstantFoldIterations limits the number of times literals may be folding during optimization.
//
// Defaults to 100 if not set.
func MaxConstantFoldIterations(limit int) ConstantFoldingOption {
	return func(opt *constantFoldingOptimizer) (*constantFoldingOptimizer, error) {
		opt.maxFoldIterations = limit
		return opt, nil
	}
}

// NewConstantFoldingOptimizer creates an optimizer which inlines constant scalar an aggregate
// literal values within function calls and select statements with their evaluated result.
func NewConstantFoldingOptimizer(opts ...ConstantFoldingOption) (ASTOptimizer, error) {
	folder := &constantFoldingOptimizer{
		maxFoldIterations: defaultMaxConstantFoldIterations,
	}
	var err error
	for _, o := range opts {
		folder, err = o(folder)
		if err != nil {
			return nil, err
		}
	}
	return folder, nil
}

type constantFoldingOptimizer struct {
	maxFoldIterations int
}

// Optimize queries the expression graph for scalar and aggregate literal expressions within call and
// select statements and then evaluates them and replaces the call site with the literal result.
//
// Note: only values which can be represented as literals in CEL syntax are supported.
func (opt *constantFoldingOptimizer) Optimize(ctx *OptimizerContext, a *ast.AST) *ast.AST {
	root := ast.NavigateAST(a)

	// Walk the list of foldable expression and continue to fold until there are no more folds left.
	// All of the fold candidates returned by the constantExprMatcher should succeed unless there's
	// a logic bug with the selection of expressions.
	foldableExprs := ast.MatchDescendants(root, constantExprMatcher)
	foldCount := 0
	for len(foldableExprs) != 0 && foldCount < opt.maxFoldIterations {
		for _, fold := range foldableExprs {
			// If the expression could be folded because it's a non-strict call, and the
			// branches are pruned, continue to the next fold.
			if fold.Kind() == ast.CallKind && maybePruneBranches(ctx, fold) {
				continue
			}
			// Otherwise, assume all context is needed to evaluate the expression.
			err := tryFold(ctx, a, fold)
			if err != nil {
				ctx.ReportErrorAtID(fold.ID(), "constant-folding evaluation failed: %v", err.Error())
				return a
			}
		}
		foldCount++
		foldableExprs = ast.MatchDescendants(root, constantExprMatcher)
	}
	// Once all of the constants have been folded, try to run through the remaining comprehensions
	// one last time. In this case, there's no guarantee they'll run, so we only update the
	// target comprehension node with the literal value if the evaluation succeeds.
	for _, compre := range ast.MatchDescendants(root, ast.KindMatcher(ast.ComprehensionKind)) {
		tryFold(ctx, a, compre)
	}

	// If the output is a list, map, or struct which contains optional entries, then prune it
	// to make sure that the optionals, if resolved, do not surface in the output literal.
	pruneOptionalElements(ctx, root)

	// Ensure that all intermediate values in the folded expression can be represented as valid
	// CEL literals within the AST structure. Use `PostOrderVisit` rather than `MatchDescendents`
	// to avoid extra allocations during this final pass through the AST.
	ast.PostOrderVisit(root, ast.NewExprVisitor(func(e ast.Expr) {
		if e.Kind() != ast.LiteralKind {
			return
		}
		val := e.AsLiteral()
		adapted, err := adaptLiteral(ctx, val)
		if err != nil {
			ctx.ReportErrorAtID(root.ID(), "constant-folding evaluation failed: %v", err.Error())
			return
		}
		ctx.UpdateExpr(e, adapted)
	}))

	return a
}

// tryFold attempts to evaluate a sub-expression to a literal.
//
// If the evaluation succeeds, the input expr value will be modified to become a literal, otherwise
// the method will return an error.
func tryFold(ctx *OptimizerContext, a *ast.AST, expr ast.Expr) error {
	// Assume all context is needed to evaluate the expression.
	subAST := &Ast{
		impl: ast.NewCheckedAST(ast.NewAST(expr, a.SourceInfo()), a.TypeMap(), a.ReferenceMap()),
	}
	prg, err := ctx.Program(subAST)
	if err != nil {
		return err
	}
	out, _, err := prg.Eval(NoVars())
	if err != nil {
		return err
	}
	// Update the fold expression to be a literal.
	ctx.UpdateExpr(expr, ctx.NewLiteral(out))
	return nil
}

// maybePruneBranches inspects the non-strict call expression to determine whether
// a branch can be removed. Evaluation will naturally prune logical and / or calls,
// but conditional will not be pruned cleanly, so this is one small area where the
// constant folding step reimplements a portion of the evaluator.
func maybePruneBranches(ctx *OptimizerContext, expr ast.NavigableExpr) bool {
	call := expr.AsCall()
	args := call.Args()
	switch call.FunctionName() {
	case operators.LogicalAnd, operators.LogicalOr:
		return maybeShortcircuitLogic(ctx, call.FunctionName(), args, expr)
	case operators.Conditional:
		cond := args[0]
		truthy := args[1]
		falsy := args[2]
		if cond.Kind() != ast.LiteralKind {
			return false
		}
		if cond.AsLiteral() == types.True {
			ctx.UpdateExpr(expr, truthy)
		} else {
			ctx.UpdateExpr(expr, falsy)
		}
		return true
	case operators.In:
		haystack := args[1]
		if haystack.Kind() == ast.ListKind && haystack.AsList().Size() == 0 {
			ctx.UpdateExpr(expr, ctx.NewLiteral(types.False))
			return true
		}
		needle := args[0]
		if needle.Kind() == ast.LiteralKind && haystack.Kind() == ast.ListKind {
			needleValue := needle.AsLiteral()
			list := haystack.AsList()
			for _, e := range list.Elements() {
				if e.Kind() == ast.LiteralKind && e.AsLiteral().Equal(needleValue) == types.True {
					ctx.UpdateExpr(expr, ctx.NewLiteral(types.True))
					return true
				}
			}
		}
	}
	return false
}

func maybeShortcircuitLogic(ctx *OptimizerContext, function string, args []ast.Expr, expr ast.NavigableExpr) bool {
	shortcircuit := types.False
	skip := types.True
	if function == operators.LogicalOr {
		shortcircuit = types.True
		skip = types.False
	}
	newArgs := []ast.Expr{}
	for _, arg := range args {
		if arg.Kind() != ast.LiteralKind {
			newArgs = append(newArgs, arg)
			continue
		}
		if arg.AsLiteral() == skip {
			continue
		}
		if arg.AsLiteral() == shortcircuit {
			ctx.UpdateExpr(expr, arg)
			return true
		}
	}
	if len(newArgs) == 0 {
		newArgs = append(newArgs, args[0])
		ctx.UpdateExpr(expr, newArgs[0])
		return true
	}
	if len(newArgs) == 1 {
		ctx.UpdateExpr(expr, newArgs[0])
		return true
	}
	ctx.UpdateExpr(expr, ctx.NewCall(function, newArgs...))
	return true
}

// pruneOptionalElements works from the bottom up to resolve optional elements within
// aggregate literals.
//
// Note, many aggregate literals will be resolved as arguments to functions or select
// statements, so this method exists to handle the case where the literal could not be
// fully resolved or exists outside of a call, select, or comprehension context.
func pruneOptionalElements(ctx *OptimizerContext, root ast.NavigableExpr) {
	aggregateLiterals := ast.MatchDescendants(root, aggregateLiteralMatcher)
	for _, lit := range aggregateLiterals {
		switch lit.Kind() {
		case ast.ListKind:
			pruneOptionalListElements(ctx, lit)
		case ast.MapKind:
			pruneOptionalMapEntries(ctx, lit)
		case ast.StructKind:
			pruneOptionalStructFields(ctx, lit)
		}
	}
}

func pruneOptionalListElements(ctx *OptimizerContext, e ast.Expr) {
	l := e.AsList()
	elems := l.Elements()
	optIndices := l.OptionalIndices()
	if len(optIndices) == 0 {
		return
	}
	updatedElems := []ast.Expr{}
	updatedIndices := []int32{}
	newOptIndex := -1
	for _, e := range elems {
		newOptIndex++
		if !l.IsOptional(int32(newOptIndex)) {
			updatedElems = append(updatedElems, e)
			continue
		}
		if e.Kind() != ast.LiteralKind {
			updatedElems = append(updatedElems, e)
			updatedIndices = append(updatedIndices, int32(newOptIndex))
			continue
		}
		optElemVal, ok := e.AsLiteral().(*types.Optional)
		if !ok {
			updatedElems = append(updatedElems, e)
			updatedIndices = append(updatedIndices, int32(newOptIndex))
			continue
		}
		if !optElemVal.HasValue() {
			newOptIndex-- // Skipping causes the list to get smaller.
			continue
		}
		ctx.UpdateExpr(e, ctx.NewLiteral(optElemVal.GetValue()))
		updatedElems = append(updatedElems, e)
	}
	ctx.UpdateExpr(e, ctx.NewList(updatedElems, updatedIndices))
}

func pruneOptionalMapEntries(ctx *OptimizerContext, e ast.Expr) {
	m := e.AsMap()
	entries := m.Entries()
	updatedEntries := []ast.EntryExpr{}
	modified := false
	for _, e := range entries {
		entry := e.AsMapEntry()
		key := entry.Key()
		val := entry.Value()
		// If the entry is not optional, or the value-side of the optional hasn't
		// been resolved to a literal, then preserve the entry as-is.
		if !entry.IsOptional() || val.Kind() != ast.LiteralKind {
			updatedEntries = append(updatedEntries, e)
			continue
		}
		optElemVal, ok := val.AsLiteral().(*types.Optional)
		if !ok {
			updatedEntries = append(updatedEntries, e)
			continue
		}
		// When the key is not a literal, but the value is, then it needs to be
		// restored to an optional value.
		if key.Kind() != ast.LiteralKind {
			undoOptVal, err := adaptLiteral(ctx, optElemVal)
			if err != nil {
				ctx.ReportErrorAtID(val.ID(), "invalid map value literal %v: %v", optElemVal, err)
			}
			ctx.UpdateExpr(val, undoOptVal)
			updatedEntries = append(updatedEntries, e)
			continue
		}
		modified = true
		if !optElemVal.HasValue() {
			continue
		}
		ctx.UpdateExpr(val, ctx.NewLiteral(optElemVal.GetValue()))
		updatedEntry := ctx.NewMapEntry(key, val, false)
		updatedEntries = append(updatedEntries, updatedEntry)
	}
	if modified {
		ctx.UpdateExpr(e, ctx.NewMap(updatedEntries))
	}
}

func pruneOptionalStructFields(ctx *OptimizerContext, e ast.Expr) {
	s := e.AsStruct()
	fields := s.Fields()
	updatedFields := []ast.EntryExpr{}
	modified := false
	for _, f := range fields {
		field := f.AsStructField()
		val := field.Value()
		if !field.IsOptional() || val.Kind() != ast.LiteralKind {
			updatedFields = append(updatedFields, f)
			continue
		}
		optElemVal, ok := val.AsLiteral().(*types.Optional)
		if !ok {
			updatedFields = append(updatedFields, f)
			continue
		}
		modified = true
		if !optElemVal.HasValue() {
			continue
		}
		ctx.UpdateExpr(val, ctx.NewLiteral(optElemVal.GetValue()))
		updatedField := ctx.NewStructField(field.Name(), val, false)
		updatedFields = append(updatedFields, updatedField)
	}
	if modified {
		ctx.UpdateExpr(e, ctx.NewStruct(s.TypeName(), updatedFields))
	}
}

// adaptLiteral converts a runtime CEL value to its equivalent literal expression.
//
// For strongly typed values, the type-provider will be used to reconstruct the fields
// which are present in the literal and their equivalent initialization values.
func adaptLiteral(ctx *OptimizerContext, val ref.Val) (ast.Expr, error) {
	switch t := val.Type().(type) {
	case *types.Type:
		switch t {
		case types.BoolType, types.BytesType, types.DoubleType, types.IntType,
			types.NullType, types.StringType, types.UintType:
			return ctx.NewLiteral(val), nil
		case types.DurationType:
			return ctx.NewCall(
				overloads.TypeConvertDuration,
				ctx.NewLiteral(val.ConvertToType(types.StringType)),
			), nil
		case types.TimestampType:
			return ctx.NewCall(
				overloads.TypeConvertTimestamp,
				ctx.NewLiteral(val.ConvertToType(types.StringType)),
			), nil
		case types.OptionalType:
			opt := val.(*types.Optional)
			if !opt.HasValue() {
				return ctx.NewCall("optional.none"), nil
			}
			target, err := adaptLiteral(ctx, opt.GetValue())
			if err != nil {
				return nil, err
			}
			return ctx.NewCall("optional.of", target), nil
		case types.TypeType:
			return ctx.NewIdent(val.(*types.Type).TypeName()), nil
		case types.ListType:
			l, ok := val.(traits.Lister)
			if !ok {
				return nil, fmt.Errorf("failed to adapt %v to literal", val)
			}
			elems := make([]ast.Expr, l.Size().(types.Int))
			idx := 0
			it := l.Iterator()
			for it.HasNext() == types.True {
				elemVal := it.Next()
				elemExpr, err := adaptLiteral(ctx, elemVal)
				if err != nil {
					return nil, err
				}
				elems[idx] = elemExpr
				idx++
			}
			return ctx.NewList(elems, []int32{}), nil
		case types.MapType:
			m, ok := val.(traits.Mapper)
			if !ok {
				return nil, fmt.Errorf("failed to adapt %v to literal", val)
			}
			entries := make([]ast.EntryExpr, m.Size().(types.Int))
			idx := 0
			it := m.Iterator()
			for it.HasNext() == types.True {
				keyVal := it.Next()
				keyExpr, err := adaptLiteral(ctx, keyVal)
				if err != nil {
					return nil, err
				}
				valVal := m.Get(keyVal)
				valExpr, err := adaptLiteral(ctx, valVal)
				if err != nil {
					return nil, err
				}
				entries[idx] = ctx.NewMapEntry(keyExpr, valExpr, false)
				idx++
			}
			return ctx.NewMap(entries), nil
		default:
			provider := ctx.CELTypeProvider()
			fields, found := provider.FindStructFieldNames(t.TypeName())
			if !found {
				return nil, fmt.Errorf("failed to adapt %v to literal", val)
			}
			tester := val.(traits.FieldTester)
			indexer := val.(traits.Indexer)
			fieldInits := []ast.EntryExpr{}
			for _, f := range fields {
				field := types.String(f)
				if tester.IsSet(field) != types.True {
					continue
				}
				fieldVal := indexer.Get(field)
				fieldExpr, err := adaptLiteral(ctx, fieldVal)
				if err != nil {
					return nil, err
				}
				fieldInits = append(fieldInits, ctx.NewStructField(f, fieldExpr, false))
			}
			return ctx.NewStruct(t.TypeName(), fieldInits), nil
		}
	}
	return nil, fmt.Errorf("failed to adapt %v to literal", val)
}

// constantExprMatcher matches calls, select statements, and comprehensions whose arguments
// are all constant scalar or aggregate literal values.
//
// Only comprehensions which are not nested are included as possible constant folds, and only
// if all variables referenced in the comprehension stack exist are only iteration or
// accumulation variables.
func constantExprMatcher(e ast.NavigableExpr) bool {
	switch e.Kind() {
	case ast.CallKind:
		return constantCallMatcher(e)
	case ast.SelectKind:
		sel := e.AsSelect() // guaranteed to be a navigable value
		return constantMatcher(sel.Operand().(ast.NavigableExpr))
	case ast.ComprehensionKind:
		if isNestedComprehension(e) {
			return false
		}
		vars := map[string]bool{}
		constantExprs := true
		visitor := ast.NewExprVisitor(func(e ast.Expr) {
			if e.Kind() == ast.ComprehensionKind {
				nested := e.AsComprehension()
				vars[nested.AccuVar()] = true
				vars[nested.IterVar()] = true
			}
			if e.Kind() == ast.IdentKind && !vars[e.AsIdent()] {
				constantExprs = false
			}
		})
		ast.PreOrderVisit(e, visitor)
		return constantExprs
	default:
		return false
	}
}

// constantCallMatcher identifies strict and non-strict calls which can be folded.
func constantCallMatcher(e ast.NavigableExpr) bool {
	call := e.AsCall()
	children := e.Children()
	fnName := call.FunctionName()
	if fnName == operators.LogicalAnd {
		for _, child := range children {
			if child.Kind() == ast.LiteralKind {
				return true
			}
		}
	}
	if fnName == operators.LogicalOr {
		for _, child := range children {
			if child.Kind() == ast.LiteralKind {
				return true
			}
		}
	}
	if fnName == operators.Conditional {
		cond := children[0]
		if cond.Kind() == ast.LiteralKind && cond.AsLiteral().Type() == types.BoolType {
			return true
		}
	}
	if fnName == operators.In {
		haystack := children[1]
		if haystack.Kind() == ast.ListKind && haystack.AsList().Size() == 0 {
			return true
		}
		needle := children[0]
		if needle.Kind() == ast.LiteralKind && haystack.Kind() == ast.ListKind {
			needleValue := needle.AsLiteral()
			list := haystack.AsList()
			for _, e := range list.Elements() {
				if e.Kind() == ast.LiteralKind && e.AsLiteral().Equal(needleValue) == types.True {
					return true
				}
			}
		}
	}
	// convert all other calls with constant arguments
	for _, child := range children {
		if !constantMatcher(child) {
			return false
		}
	}
	return true
}

func isNestedComprehension(e ast.NavigableExpr) bool {
	parent, found := e.Parent()
	for found {
		if parent.Kind() == ast.ComprehensionKind {
			return true
		}
		parent, found = parent.Parent()
	}
	return false
}

func aggregateLiteralMatcher(e ast.NavigableExpr) bool {
	return e.Kind() == ast.ListKind || e.Kind() == ast.MapKind || e.Kind() == ast.StructKind
}

var (
	constantMatcher = ast.ConstantValueMatcher()
)

const (
	defaultMaxConstantFoldIterations = 100
)
