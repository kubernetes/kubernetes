package sharedcheck

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"honnef.co/go/tools/analysis/code"
	"honnef.co/go/tools/analysis/edit"
	"honnef.co/go/tools/analysis/facts"
	"honnef.co/go/tools/analysis/report"
	"honnef.co/go/tools/go/ast/astutil"
	"honnef.co/go/tools/go/ir"
	"honnef.co/go/tools/go/ir/irutil"
	"honnef.co/go/tools/internal/passes/buildir"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
)

func CheckRangeStringRunes(pass *analysis.Pass) (interface{}, error) {
	for _, fn := range pass.ResultOf[buildir.Analyzer].(*buildir.IR).SrcFuncs {
		cb := func(node ast.Node) bool {
			rng, ok := node.(*ast.RangeStmt)
			if !ok || !astutil.IsBlank(rng.Key) {
				return true
			}

			v, _ := fn.ValueForExpr(rng.X)

			// Check that we're converting from string to []rune
			val, _ := v.(*ir.Convert)
			if val == nil {
				return true
			}
			Tsrc, ok := val.X.Type().Underlying().(*types.Basic)
			if !ok || Tsrc.Kind() != types.String {
				return true
			}
			Tdst, ok := val.Type().(*types.Slice)
			if !ok {
				return true
			}
			TdstElem, ok := Tdst.Elem().(*types.Basic)
			if !ok || TdstElem.Kind() != types.Int32 {
				return true
			}

			// Check that the result of the conversion is only used to
			// range over
			refs := val.Referrers()
			if refs == nil {
				return true
			}

			// Expect two refs: one for obtaining the length of the slice,
			// one for accessing the elements
			if len(irutil.FilterDebug(*refs)) != 2 {
				// TODO(dh): right now, we check that only one place
				// refers to our slice. This will miss cases such as
				// ranging over the slice twice. Ideally, we'd ensure that
				// the slice is only used for ranging over (without
				// accessing the key), but that is harder to do because in
				// IR form, ranging over a slice looks like an ordinary
				// loop with index increments and slice accesses. We'd
				// have to look at the associated AST node to check that
				// it's a range statement.
				return true
			}

			pass.Reportf(rng.Pos(), "should range over string, not []rune(string)")

			return true
		}
		if source := fn.Source(); source != nil {
			ast.Inspect(source, cb)
		}
	}
	return nil, nil
}

// RedundantTypeInDeclarationChecker returns a checker that flags variable declarations with redundantly specified types.
// That is, it flags 'var v T = e' where e's type is identical to T and 'var v = e' (or 'v := e') would have the same effect.
//
// It does not flag variables under the following conditions, to reduce the number of false positives:
// - global variables – these often specify types to aid godoc
// - files that use cgo – cgo code generation and pointer checking emits redundant types
//
// It does not flag variables under the following conditions, unless flagHelpfulTypes is true, to reduce the number of noisy positives:
// - packages that import syscall or unsafe – these sometimes use this form of assignment to make sure types are as expected
// - variables named the blank identifier – a pattern used to confirm the types of variables
// - named untyped constants on the rhs – the explicitness might aid readability
func RedundantTypeInDeclarationChecker(verb string, flagHelpfulTypes bool) *analysis.Analyzer {
	fn := func(pass *analysis.Pass) (interface{}, error) {
		eval := func(expr ast.Expr) (types.TypeAndValue, error) {
			info := &types.Info{
				Types: map[ast.Expr]types.TypeAndValue{},
			}
			err := types.CheckExpr(pass.Fset, pass.Pkg, expr.Pos(), expr, info)
			return info.Types[expr], err
		}

		if !flagHelpfulTypes {
			// Don't look at code in low-level packages
			for _, imp := range pass.Pkg.Imports() {
				if imp.Path() == "syscall" || imp.Path() == "unsafe" {
					return nil, nil
				}
			}
		}

		fn := func(node ast.Node) {
			decl := node.(*ast.GenDecl)
			if decl.Tok != token.VAR {
				return
			}

			gen, _ := code.Generator(pass, decl.Pos())
			if gen == facts.Cgo {
				// TODO(dh): remove this exception once we can use UsesCgo
				return
			}

			// Delay looking up parent AST nodes until we have to
			checkedDecl := false

		specLoop:
			for _, spec := range decl.Specs {
				spec := spec.(*ast.ValueSpec)
				if spec.Type == nil {
					continue
				}
				if len(spec.Names) != len(spec.Values) {
					continue
				}
				Tlhs := pass.TypesInfo.TypeOf(spec.Type)
				for i, v := range spec.Values {
					if !flagHelpfulTypes && spec.Names[i].Name == "_" {
						continue specLoop
					}
					Trhs := pass.TypesInfo.TypeOf(v)
					if !types.Identical(Tlhs, Trhs) {
						continue specLoop
					}

					// Some expressions are untyped and get converted to the lhs type implicitly.
					// This applies to untyped constants, shift operations with an untyped lhs, and possibly others.
					//
					// Check if the type is truly redundant, i.e. if the type on the lhs doesn't match the default type of the untyped constant.
					tv, err := eval(v)
					if err != nil {
						panic(err)
					}
					if b, ok := tv.Type.(*types.Basic); ok && (b.Info()&types.IsUntyped) != 0 {
						switch v := v.(type) {
						case *ast.Ident:
							// Only flag named constant rhs if it's a predeclared identifier.
							// Don't flag other named constants, as the explicit type may aid readability.
							if pass.TypesInfo.ObjectOf(v).Pkg() != nil && !flagHelpfulTypes {
								continue specLoop
							}
						case *ast.SelectorExpr:
							// Constant selector expressions can only refer to named constants that arent predeclared.
							if !flagHelpfulTypes {
								continue specLoop
							}
						default:
							// don't skip if the type on the lhs matches the default type of the constant
							if Tlhs != types.Default(b) {
								continue specLoop
							}
						}
					}
				}

				if !checkedDecl {
					// Don't flag global variables. These often have explicit types for godoc's sake.
					path, _ := astutil.PathEnclosingInterval(code.File(pass, decl), decl.Pos(), decl.Pos())
				pathLoop:
					for _, el := range path {
						switch el.(type) {
						case *ast.FuncDecl, *ast.FuncLit:
							checkedDecl = true
							break pathLoop
						}
					}
					if !checkedDecl {
						// decl is not inside a function
						break specLoop
					}
				}

				report.Report(pass, spec.Type, fmt.Sprintf("%s omit type %s from declaration; it will be inferred from the right-hand side", verb, report.Render(pass, spec.Type)), report.FilterGenerated(),
					report.Fixes(edit.Fix("Remove redundant type", edit.Delete(spec.Type))))
			}
		}
		code.Preorder(pass, fn, (*ast.GenDecl)(nil))
		return nil, nil
	}

	return &analysis.Analyzer{
		Run:      fn,
		Requires: []*analysis.Analyzer{facts.Generated, inspect.Analyzer, facts.TokenFile, facts.Generated},
	}
}
