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
	"github.com/google/cel-go/common/ast"
	"github.com/google/cel-go/common/containers"
	"github.com/google/cel-go/common/operators"
	"github.com/google/cel-go/common/overloads"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/traits"
)

// InlineVariable holds a variable name to be matched and an AST representing
// the expression graph which should be used to replace it.
type InlineVariable struct {
	name  string
	alias string
	def   *ast.AST
}

// Name returns the qualified variable or field selection to replace.
func (v *InlineVariable) Name() string {
	return v.name
}

// Alias returns the alias to use when performing cel.bind() calls during inlining.
func (v *InlineVariable) Alias() string {
	return v.alias
}

// Expr returns the inlined expression value.
func (v *InlineVariable) Expr() ast.Expr {
	return v.def.Expr()
}

// Type indicates the inlined expression type.
func (v *InlineVariable) Type() *Type {
	return v.def.GetType(v.def.Expr().ID())
}

// NewInlineVariable declares a variable name to be replaced by a checked expression.
func NewInlineVariable(name string, definition *Ast) *InlineVariable {
	return NewInlineVariableWithAlias(name, name, definition)
}

// NewInlineVariableWithAlias declares a variable name to be replaced by a checked expression.
// If the variable occurs more than once, the provided alias will be used to replace the expressions
// where the variable name occurs.
func NewInlineVariableWithAlias(name, alias string, definition *Ast) *InlineVariable {
	return &InlineVariable{name: name, alias: alias, def: definition.impl}
}

// NewInliningOptimizer creates and optimizer which replaces variables with expression definitions.
//
// If a variable occurs one time, the variable is replaced by the inline definition. If the
// variable occurs more than once, the variable occurences are replaced by a cel.bind() call.
func NewInliningOptimizer(inlineVars ...*InlineVariable) ASTOptimizer {
	return &inliningOptimizer{variables: inlineVars}
}

type inliningOptimizer struct {
	variables []*InlineVariable
}

func (opt *inliningOptimizer) Optimize(ctx *OptimizerContext, a *ast.AST) *ast.AST {
	root := ast.NavigateAST(a)
	for _, inlineVar := range opt.variables {
		matches := ast.MatchDescendants(root, opt.matchVariable(inlineVar.Name()))
		// Skip cases where the variable isn't in the expression graph
		if len(matches) == 0 {
			continue
		}

		// For a single match, do a direct replacement of the expression sub-graph.
		if len(matches) == 1 || !isBindable(matches, inlineVar.Expr(), inlineVar.Type()) {
			for _, match := range matches {
				// Copy the inlined AST expr and source info.
				copyExpr := ctx.CopyASTAndMetadata(inlineVar.def)
				opt.inlineExpr(ctx, match, copyExpr, inlineVar.Type())
			}
			continue
		}

		// For multiple matches, find the least common ancestor (lca) and insert the
		// variable as a cel.bind() macro.
		var lca ast.NavigableExpr = root
		lcaAncestorCount := 0
		ancestors := map[int64]int{}
		for _, match := range matches {
			// Update the identifier matches with the provided alias.
			parent, found := match, true
			for found {
				ancestorCount, hasAncestor := ancestors[parent.ID()]
				if !hasAncestor {
					ancestors[parent.ID()] = 1
					parent, found = parent.Parent()
					continue
				}
				if lcaAncestorCount < ancestorCount || (lcaAncestorCount == ancestorCount && lca.Depth() < parent.Depth()) {
					lca = parent
					lcaAncestorCount = ancestorCount
				}
				ancestors[parent.ID()] = ancestorCount + 1
				parent, found = parent.Parent()
			}
			aliasExpr := ctx.NewIdent(inlineVar.Alias())
			opt.inlineExpr(ctx, match, aliasExpr, inlineVar.Type())
		}

		// Copy the inlined AST expr and source info.
		copyExpr := ctx.CopyASTAndMetadata(inlineVar.def)
		// Update the least common ancestor by inserting a cel.bind() call to the alias.
		inlined, bindMacro := ctx.NewBindMacro(lca.ID(), inlineVar.Alias(), copyExpr, lca)
		opt.inlineExpr(ctx, lca, inlined, inlineVar.Type())
		ctx.SetMacroCall(lca.ID(), bindMacro)
	}
	return a
}

// inlineExpr replaces the current expression with the inlined one, unless the location of the inlining
// happens within a presence test, e.g. has(a.b.c) -> inline alpha for a.b.c in which case an attempt is
// made to determine whether the inlined value can be presence or existence tested.
func (opt *inliningOptimizer) inlineExpr(ctx *OptimizerContext, prev ast.NavigableExpr, inlined ast.Expr, inlinedType *Type) {
	switch prev.Kind() {
	case ast.SelectKind:
		sel := prev.AsSelect()
		if !sel.IsTestOnly() {
			ctx.UpdateExpr(prev, inlined)
			return
		}
		opt.rewritePresenceExpr(ctx, prev, inlined, inlinedType)
	default:
		ctx.UpdateExpr(prev, inlined)
	}
}

// rewritePresenceExpr converts the inlined expression, when it occurs within a has() macro, to type-safe
// expression appropriate for the inlined type, if possible.
//
// If the rewrite is not possible an error is reported at the inline expression site.
func (opt *inliningOptimizer) rewritePresenceExpr(ctx *OptimizerContext, prev, inlined ast.Expr, inlinedType *Type) {
	// If the input inlined expression is not a select expression it won't work with the has()
	// macro. Attempt to rewrite the presence test in terms of the typed input, otherwise error.
	if inlined.Kind() == ast.SelectKind {
		presenceTest, hasMacro := ctx.NewHasMacro(prev.ID(), inlined)
		ctx.UpdateExpr(prev, presenceTest)
		ctx.SetMacroCall(prev.ID(), hasMacro)
		return
	}

	ctx.ClearMacroCall(prev.ID())
	if inlinedType.IsAssignableType(NullType) {
		ctx.UpdateExpr(prev,
			ctx.NewCall(operators.NotEquals,
				inlined,
				ctx.NewLiteral(types.NullValue),
			))
		return
	}
	if inlinedType.HasTrait(traits.SizerType) {
		ctx.UpdateExpr(prev,
			ctx.NewCall(operators.NotEquals,
				ctx.NewMemberCall(overloads.Size, inlined),
				ctx.NewLiteral(types.IntZero),
			))
		return
	}
	ctx.ReportErrorAtID(prev.ID(), "unable to inline expression type %v into presence test", inlinedType)
}

// isBindable indicates whether the inlined type can be used within a cel.bind() if the expression
// being replaced occurs within a presence test. Value types with a size() method or field selection
// support can be bound.
//
// In future iterations, support may also be added for indexer types which can be rewritten as an `in`
// expression; however, this would imply a rewrite of the inlined expression that may not be necessary
// in most cases.
func isBindable(matches []ast.NavigableExpr, inlined ast.Expr, inlinedType *Type) bool {
	if inlinedType.IsAssignableType(NullType) ||
		inlinedType.HasTrait(traits.SizerType) {
		return true
	}
	for _, m := range matches {
		if m.Kind() != ast.SelectKind {
			continue
		}
		sel := m.AsSelect()
		if sel.IsTestOnly() {
			return false
		}
	}
	return true
}

// matchVariable matches simple identifiers, select expressions, and presence test expressions
// which match the (potentially) qualified variable name provided as input.
//
// Note, this function does not support inlining against select expressions which includes optional
// field selection. This may be a future refinement.
func (opt *inliningOptimizer) matchVariable(varName string) ast.ExprMatcher {
	return func(e ast.NavigableExpr) bool {
		if e.Kind() == ast.IdentKind && e.AsIdent() == varName {
			return true
		}
		if e.Kind() == ast.SelectKind {
			sel := e.AsSelect()
			// While the `ToQualifiedName` call could take the select directly, this
			// would skip presence tests from possible matches, which we would like
			// to include.
			qualName, found := containers.ToQualifiedName(sel.Operand())
			return found && qualName+"."+sel.FieldName() == varName
		}
		return false
	}
}
