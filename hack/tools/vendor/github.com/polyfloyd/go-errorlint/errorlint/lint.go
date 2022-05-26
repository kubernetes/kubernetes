package errorlint

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"regexp"
)

type Lint struct {
	Message string
	Pos     token.Pos
}

type ByPosition []Lint

func (l ByPosition) Len() int      { return len(l) }
func (l ByPosition) Swap(i, j int) { l[i], l[j] = l[j], l[i] }

func (l ByPosition) Less(i, j int) bool {
	return l[i].Pos < l[j].Pos
}

func LintFmtErrorfCalls(fset *token.FileSet, info types.Info) []Lint {
	lints := []Lint{}
	for expr, t := range info.Types {
		// Search for error expressions that are the result of fmt.Errorf
		// invocations.
		if t.Type.String() != "error" {
			continue
		}
		call, ok := isFmtErrorfCallExpr(info, expr)
		if !ok {
			continue
		}

		// Find all % fields in the format string.
		formatVerbs, ok := printfFormatStringVerbs(info, call)
		if !ok {
			continue
		}

		// For any arguments that are errors, check whether the wrapping verb
		// is used. Only one %w verb may be used in a single format string at a
		// time, so we stop after finding a correct %w.
		var lintArg ast.Expr
		args := call.Args[1:]
		for i := 0; i < len(args) && i < len(formatVerbs); i++ {
			if !implementsError(info.Types[args[i]].Type) && !isErrorStringCall(info, args[i]) {
				continue
			}

			if formatVerbs[i] == "%w" {
				lintArg = nil
				break
			}

			if lintArg == nil {
				lintArg = args[i]
			}
		}
		if lintArg != nil {
			lints = append(lints, Lint{
				Message: "non-wrapping format verb for fmt.Errorf. Use `%w` to format errors",
				Pos:     lintArg.Pos(),
			})
		}
	}
	return lints
}

// isErrorStringCall tests whether the expression is a string expression that
// is the result of an `(error).Error()` method call.
func isErrorStringCall(info types.Info, expr ast.Expr) bool {
	if info.Types[expr].Type.String() == "string" {
		if call, ok := expr.(*ast.CallExpr); ok {
			if callSel, ok := call.Fun.(*ast.SelectorExpr); ok {
				fun := info.Uses[callSel.Sel].(*types.Func)
				return fun.Type().String() == "func() string" && fun.Name() == "Error"
			}
		}
	}
	return false
}

func printfFormatStringVerbs(info types.Info, call *ast.CallExpr) ([]string, bool) {
	if len(call.Args) <= 1 {
		return nil, false
	}
	strLit, ok := call.Args[0].(*ast.BasicLit)
	if !ok {
		// Ignore format strings that are not literals.
		return nil, false
	}
	formatString := constant.StringVal(info.Types[strLit].Value)

	// Naive format string argument verb. This does not take modifiers such as
	// padding into account...
	re := regexp.MustCompile(`%[^%]`)
	return re.FindAllString(formatString, -1), true
}

func isFmtErrorfCallExpr(info types.Info, expr ast.Expr) (*ast.CallExpr, bool) {
	call, ok := expr.(*ast.CallExpr)
	if !ok {
		return nil, false
	}
	fn, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		// TODO: Support fmt.Errorf variable aliases?
		return nil, false
	}
	obj := info.Uses[fn.Sel]

	pkg := obj.Pkg()
	if pkg != nil && pkg.Name() == "fmt" && obj.Name() == "Errorf" {
		return call, true
	}
	return nil, false
}

func LintErrorComparisons(fset *token.FileSet, info *TypesInfoExt) []Lint {
	lints := []Lint{}

	for expr := range info.Types {
		// Find == and != operations.
		binExpr, ok := expr.(*ast.BinaryExpr)
		if !ok {
			continue
		}
		if binExpr.Op != token.EQL && binExpr.Op != token.NEQ {
			continue
		}
		// Comparing errors with nil is okay.
		if isNilComparison(binExpr) {
			continue
		}
		// Find comparisons of which one side is a of type error.
		if !isErrorComparison(info.Info, binExpr) {
			continue
		}

		if isAllowedErrorComparison(info, binExpr) {
			continue
		}

		lints = append(lints, Lint{
			Message: fmt.Sprintf("comparing with %s will fail on wrapped errors. Use errors.Is to check for a specific error", binExpr.Op),
			Pos:     binExpr.Pos(),
		})
	}

	for scope := range info.Scopes {
		// Find value switch blocks.
		switchStmt, ok := scope.(*ast.SwitchStmt)
		if !ok {
			continue
		}
		// Check whether the switch operates on an error type.
		if switchStmt.Tag == nil {
			continue
		}
		tagType := info.Types[switchStmt.Tag]
		if tagType.Type.String() != "error" {
			continue
		}

		lints = append(lints, Lint{
			Message: "switch on an error will fail on wrapped errors. Use errors.Is to check for specific errors",
			Pos:     switchStmt.Pos(),
		})
	}

	return lints
}

func isNilComparison(binExpr *ast.BinaryExpr) bool {
	if ident, ok := binExpr.X.(*ast.Ident); ok && ident.Name == "nil" {
		return true
	}
	if ident, ok := binExpr.Y.(*ast.Ident); ok && ident.Name == "nil" {
		return true
	}
	return false
}

func isErrorComparison(info types.Info, binExpr *ast.BinaryExpr) bool {
	tx := info.Types[binExpr.X]
	ty := info.Types[binExpr.Y]
	return tx.Type.String() == "error" || ty.Type.String() == "error"
}

func LintErrorTypeAssertions(fset *token.FileSet, info types.Info) []Lint {
	lints := []Lint{}

	for expr := range info.Types {
		// Find type assertions.
		typeAssert, ok := expr.(*ast.TypeAssertExpr)
		if !ok {
			continue
		}

		// Find type assertions that operate on values of type error.
		if !isErrorTypeAssertion(info, typeAssert) {
			continue
		}

		lints = append(lints, Lint{
			Message: "type assertion on error will fail on wrapped errors. Use errors.As to check for specific errors",
			Pos:     typeAssert.Pos(),
		})
	}

	for scope := range info.Scopes {
		// Find type switches.
		typeSwitch, ok := scope.(*ast.TypeSwitchStmt)
		if !ok {
			continue
		}

		// Find the type assertion in the type switch.
		var typeAssert *ast.TypeAssertExpr
		switch t := typeSwitch.Assign.(type) {
		case *ast.ExprStmt:
			typeAssert = t.X.(*ast.TypeAssertExpr)
		case *ast.AssignStmt:
			typeAssert = t.Rhs[0].(*ast.TypeAssertExpr)
		}

		// Check whether the type switch is on a value of type error.
		if !isErrorTypeAssertion(info, typeAssert) {
			continue
		}

		lints = append(lints, Lint{
			Message: "type switch on error will fail on wrapped errors. Use errors.As to check for specific errors",
			Pos:     typeAssert.Pos(),
		})
	}

	return lints
}

func isErrorTypeAssertion(info types.Info, typeAssert *ast.TypeAssertExpr) bool {
	t := info.Types[typeAssert.X]
	return t.Type.String() == "error"
}

func implementsError(t types.Type) bool {
	mset := types.NewMethodSet(t)

	for i := 0; i < mset.Len(); i++ {
		if mset.At(i).Kind() != types.MethodVal {
			continue
		}

		obj := mset.At(i).Obj()
		if obj.Name() == "Error" && obj.Type().String() == "func() string" {
			return true
		}
	}

	return false
}
