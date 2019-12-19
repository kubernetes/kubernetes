// Package simple contains a linter for Go source code.
package simple // import "honnef.co/go/tools/simple"

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"reflect"
	"sort"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/types/typeutil"
	. "honnef.co/go/tools/arg"
	"honnef.co/go/tools/internal/passes/buildssa"
	"honnef.co/go/tools/internal/sharedcheck"
	"honnef.co/go/tools/lint"
	. "honnef.co/go/tools/lint/lintdsl"
)

func LintSingleCaseSelect(pass *analysis.Pass) (interface{}, error) {
	isSingleSelect := func(node ast.Node) bool {
		v, ok := node.(*ast.SelectStmt)
		if !ok {
			return false
		}
		return len(v.Body.List) == 1
	}

	seen := map[ast.Node]struct{}{}
	fn := func(node ast.Node) {
		switch v := node.(type) {
		case *ast.ForStmt:
			if len(v.Body.List) != 1 {
				return
			}
			if !isSingleSelect(v.Body.List[0]) {
				return
			}
			if _, ok := v.Body.List[0].(*ast.SelectStmt).Body.List[0].(*ast.CommClause).Comm.(*ast.SendStmt); ok {
				// Don't suggest using range for channel sends
				return
			}
			seen[v.Body.List[0]] = struct{}{}
			ReportNodefFG(pass, node, "should use for range instead of for { select {} }")
		case *ast.SelectStmt:
			if _, ok := seen[v]; ok {
				return
			}
			if !isSingleSelect(v) {
				return
			}
			ReportNodefFG(pass, node, "should use a simple channel send/receive instead of select with a single case")
		}
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.ForStmt)(nil), (*ast.SelectStmt)(nil)}, fn)
	return nil, nil
}

func LintLoopCopy(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		loop := node.(*ast.RangeStmt)

		if loop.Key == nil {
			return
		}
		if len(loop.Body.List) != 1 {
			return
		}
		stmt, ok := loop.Body.List[0].(*ast.AssignStmt)
		if !ok {
			return
		}
		if stmt.Tok != token.ASSIGN || len(stmt.Lhs) != 1 || len(stmt.Rhs) != 1 {
			return
		}
		lhs, ok := stmt.Lhs[0].(*ast.IndexExpr)
		if !ok {
			return
		}

		if _, ok := pass.TypesInfo.TypeOf(lhs.X).(*types.Slice); !ok {
			return
		}
		lidx, ok := lhs.Index.(*ast.Ident)
		if !ok {
			return
		}
		key, ok := loop.Key.(*ast.Ident)
		if !ok {
			return
		}
		if pass.TypesInfo.TypeOf(lhs) == nil || pass.TypesInfo.TypeOf(stmt.Rhs[0]) == nil {
			return
		}
		if pass.TypesInfo.ObjectOf(lidx) != pass.TypesInfo.ObjectOf(key) {
			return
		}
		if !types.Identical(pass.TypesInfo.TypeOf(lhs), pass.TypesInfo.TypeOf(stmt.Rhs[0])) {
			return
		}
		if _, ok := pass.TypesInfo.TypeOf(loop.X).(*types.Slice); !ok {
			return
		}

		if rhs, ok := stmt.Rhs[0].(*ast.IndexExpr); ok {
			rx, ok := rhs.X.(*ast.Ident)
			_ = rx
			if !ok {
				return
			}
			ridx, ok := rhs.Index.(*ast.Ident)
			if !ok {
				return
			}
			if pass.TypesInfo.ObjectOf(ridx) != pass.TypesInfo.ObjectOf(key) {
				return
			}
		} else if rhs, ok := stmt.Rhs[0].(*ast.Ident); ok {
			value, ok := loop.Value.(*ast.Ident)
			if !ok {
				return
			}
			if pass.TypesInfo.ObjectOf(rhs) != pass.TypesInfo.ObjectOf(value) {
				return
			}
		} else {
			return
		}
		ReportNodefFG(pass, loop, "should use copy() instead of a loop")
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.RangeStmt)(nil)}, fn)
	return nil, nil
}

func LintIfBoolCmp(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		expr := node.(*ast.BinaryExpr)
		if expr.Op != token.EQL && expr.Op != token.NEQ {
			return
		}
		x := IsBoolConst(pass, expr.X)
		y := IsBoolConst(pass, expr.Y)
		if !x && !y {
			return
		}
		var other ast.Expr
		var val bool
		if x {
			val = BoolConst(pass, expr.X)
			other = expr.Y
		} else {
			val = BoolConst(pass, expr.Y)
			other = expr.X
		}
		basic, ok := pass.TypesInfo.TypeOf(other).Underlying().(*types.Basic)
		if !ok || basic.Kind() != types.Bool {
			return
		}
		op := ""
		if (expr.Op == token.EQL && !val) || (expr.Op == token.NEQ && val) {
			op = "!"
		}
		r := op + Render(pass, other)
		l1 := len(r)
		r = strings.TrimLeft(r, "!")
		if (l1-len(r))%2 == 1 {
			r = "!" + r
		}
		if IsInTest(pass, node) {
			return
		}
		ReportNodefFG(pass, expr, "should omit comparison to bool constant, can be simplified to %s", r)
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.BinaryExpr)(nil)}, fn)
	return nil, nil
}

func LintBytesBufferConversions(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		if len(call.Args) != 1 {
			return
		}

		argCall, ok := call.Args[0].(*ast.CallExpr)
		if !ok {
			return
		}
		sel, ok := argCall.Fun.(*ast.SelectorExpr)
		if !ok {
			return
		}

		typ := pass.TypesInfo.TypeOf(call.Fun)
		if typ == types.Universe.Lookup("string").Type() && IsCallToAST(pass, call.Args[0], "(*bytes.Buffer).Bytes") {
			ReportNodefFG(pass, call, "should use %v.String() instead of %v", Render(pass, sel.X), Render(pass, call))
		} else if typ, ok := typ.(*types.Slice); ok && typ.Elem() == types.Universe.Lookup("byte").Type() && IsCallToAST(pass, call.Args[0], "(*bytes.Buffer).String") {
			ReportNodefFG(pass, call, "should use %v.Bytes() instead of %v", Render(pass, sel.X), Render(pass, call))
		}

	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.CallExpr)(nil)}, fn)
	return nil, nil
}

func LintStringsContains(pass *analysis.Pass) (interface{}, error) {
	// map of value to token to bool value
	allowed := map[int64]map[token.Token]bool{
		-1: {token.GTR: true, token.NEQ: true, token.EQL: false},
		0:  {token.GEQ: true, token.LSS: false},
	}
	fn := func(node ast.Node) {
		expr := node.(*ast.BinaryExpr)
		switch expr.Op {
		case token.GEQ, token.GTR, token.NEQ, token.LSS, token.EQL:
		default:
			return
		}

		value, ok := ExprToInt(pass, expr.Y)
		if !ok {
			return
		}

		allowedOps, ok := allowed[value]
		if !ok {
			return
		}
		b, ok := allowedOps[expr.Op]
		if !ok {
			return
		}

		call, ok := expr.X.(*ast.CallExpr)
		if !ok {
			return
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return
		}
		pkgIdent, ok := sel.X.(*ast.Ident)
		if !ok {
			return
		}
		funIdent := sel.Sel
		if pkgIdent.Name != "strings" && pkgIdent.Name != "bytes" {
			return
		}
		newFunc := ""
		switch funIdent.Name {
		case "IndexRune":
			newFunc = "ContainsRune"
		case "IndexAny":
			newFunc = "ContainsAny"
		case "Index":
			newFunc = "Contains"
		default:
			return
		}

		prefix := ""
		if !b {
			prefix = "!"
		}
		ReportNodefFG(pass, node, "should use %s%s.%s(%s) instead", prefix, pkgIdent.Name, newFunc, RenderArgs(pass, call.Args))
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.BinaryExpr)(nil)}, fn)
	return nil, nil
}

func LintBytesCompare(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		expr := node.(*ast.BinaryExpr)
		if expr.Op != token.NEQ && expr.Op != token.EQL {
			return
		}
		call, ok := expr.X.(*ast.CallExpr)
		if !ok {
			return
		}
		if !IsCallToAST(pass, call, "bytes.Compare") {
			return
		}
		value, ok := ExprToInt(pass, expr.Y)
		if !ok || value != 0 {
			return
		}
		args := RenderArgs(pass, call.Args)
		prefix := ""
		if expr.Op == token.NEQ {
			prefix = "!"
		}
		ReportNodefFG(pass, node, "should use %sbytes.Equal(%s) instead", prefix, args)
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.BinaryExpr)(nil)}, fn)
	return nil, nil
}

func LintForTrue(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		loop := node.(*ast.ForStmt)
		if loop.Init != nil || loop.Post != nil {
			return
		}
		if !IsBoolConst(pass, loop.Cond) || !BoolConst(pass, loop.Cond) {
			return
		}
		ReportNodefFG(pass, loop, "should use for {} instead of for true {}")
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.ForStmt)(nil)}, fn)
	return nil, nil
}

func LintRegexpRaw(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		if !IsCallToAST(pass, call, "regexp.MustCompile") &&
			!IsCallToAST(pass, call, "regexp.Compile") {
			return
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if len(call.Args) != 1 {
			// invalid function call
			return
		}
		lit, ok := call.Args[Arg("regexp.Compile.expr")].(*ast.BasicLit)
		if !ok {
			// TODO(dominikh): support string concat, maybe support constants
			return
		}
		if lit.Kind != token.STRING {
			// invalid function call
			return
		}
		if lit.Value[0] != '"' {
			// already a raw string
			return
		}
		val := lit.Value
		if !strings.Contains(val, `\\`) {
			return
		}
		if strings.Contains(val, "`") {
			return
		}

		bs := false
		for _, c := range val {
			if !bs && c == '\\' {
				bs = true
				continue
			}
			if bs && c == '\\' {
				bs = false
				continue
			}
			if bs {
				// backslash followed by non-backslash -> escape sequence
				return
			}
		}

		ReportNodefFG(pass, call, "should use raw string (`...`) with regexp.%s to avoid having to escape twice", sel.Sel.Name)
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.CallExpr)(nil)}, fn)
	return nil, nil
}

func LintIfReturn(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		block := node.(*ast.BlockStmt)
		l := len(block.List)
		if l < 2 {
			return
		}
		n1, n2 := block.List[l-2], block.List[l-1]

		if len(block.List) >= 3 {
			if _, ok := block.List[l-3].(*ast.IfStmt); ok {
				// Do not flag a series of if statements
				return
			}
		}
		// if statement with no init, no else, a single condition
		// checking an identifier or function call and just a return
		// statement in the body, that returns a boolean constant
		ifs, ok := n1.(*ast.IfStmt)
		if !ok {
			return
		}
		if ifs.Else != nil || ifs.Init != nil {
			return
		}
		if len(ifs.Body.List) != 1 {
			return
		}
		if op, ok := ifs.Cond.(*ast.BinaryExpr); ok {
			switch op.Op {
			case token.EQL, token.LSS, token.GTR, token.NEQ, token.LEQ, token.GEQ:
			default:
				return
			}
		}
		ret1, ok := ifs.Body.List[0].(*ast.ReturnStmt)
		if !ok {
			return
		}
		if len(ret1.Results) != 1 {
			return
		}
		if !IsBoolConst(pass, ret1.Results[0]) {
			return
		}

		ret2, ok := n2.(*ast.ReturnStmt)
		if !ok {
			return
		}
		if len(ret2.Results) != 1 {
			return
		}
		if !IsBoolConst(pass, ret2.Results[0]) {
			return
		}

		if ret1.Results[0].(*ast.Ident).Name == ret2.Results[0].(*ast.Ident).Name {
			// we want the function to return true and false, not the
			// same value both times.
			return
		}

		ReportNodefFG(pass, n1, "should use 'return <expr>' instead of 'if <expr> { return <bool> }; return <bool>'")
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.BlockStmt)(nil)}, fn)
	return nil, nil
}

// LintRedundantNilCheckWithLen checks for the following reduntant nil-checks:
//
//   if x == nil || len(x) == 0 {}
//   if x != nil && len(x) != 0 {}
//   if x != nil && len(x) == N {} (where N != 0)
//   if x != nil && len(x) > N {}
//   if x != nil && len(x) >= N {} (where N != 0)
//
func LintRedundantNilCheckWithLen(pass *analysis.Pass) (interface{}, error) {
	isConstZero := func(expr ast.Expr) (isConst bool, isZero bool) {
		_, ok := expr.(*ast.BasicLit)
		if ok {
			return true, IsZero(expr)
		}
		id, ok := expr.(*ast.Ident)
		if !ok {
			return false, false
		}
		c, ok := pass.TypesInfo.ObjectOf(id).(*types.Const)
		if !ok {
			return false, false
		}
		return true, c.Val().Kind() == constant.Int && c.Val().String() == "0"
	}

	fn := func(node ast.Node) {
		// check that expr is "x || y" or "x && y"
		expr := node.(*ast.BinaryExpr)
		if expr.Op != token.LOR && expr.Op != token.LAND {
			return
		}
		eqNil := expr.Op == token.LOR

		// check that x is "xx == nil" or "xx != nil"
		x, ok := expr.X.(*ast.BinaryExpr)
		if !ok {
			return
		}
		if eqNil && x.Op != token.EQL {
			return
		}
		if !eqNil && x.Op != token.NEQ {
			return
		}
		xx, ok := x.X.(*ast.Ident)
		if !ok {
			return
		}
		if !IsNil(pass, x.Y) {
			return
		}

		// check that y is "len(xx) == 0" or "len(xx) ... "
		y, ok := expr.Y.(*ast.BinaryExpr)
		if !ok {
			return
		}
		if eqNil && y.Op != token.EQL { // must be len(xx) *==* 0
			return
		}
		yx, ok := y.X.(*ast.CallExpr)
		if !ok {
			return
		}
		yxFun, ok := yx.Fun.(*ast.Ident)
		if !ok || yxFun.Name != "len" || len(yx.Args) != 1 {
			return
		}
		yxArg, ok := yx.Args[Arg("len.v")].(*ast.Ident)
		if !ok {
			return
		}
		if yxArg.Name != xx.Name {
			return
		}

		if eqNil && !IsZero(y.Y) { // must be len(x) == *0*
			return
		}

		if !eqNil {
			isConst, isZero := isConstZero(y.Y)
			if !isConst {
				return
			}
			switch y.Op {
			case token.EQL:
				// avoid false positive for "xx != nil && len(xx) == 0"
				if isZero {
					return
				}
			case token.GEQ:
				// avoid false positive for "xx != nil && len(xx) >= 0"
				if isZero {
					return
				}
			case token.NEQ:
				// avoid false positive for "xx != nil && len(xx) != <non-zero>"
				if !isZero {
					return
				}
			case token.GTR:
				// ok
			default:
				return
			}
		}

		// finally check that xx type is one of array, slice, map or chan
		// this is to prevent false positive in case if xx is a pointer to an array
		var nilType string
		switch pass.TypesInfo.TypeOf(xx).(type) {
		case *types.Slice:
			nilType = "nil slices"
		case *types.Map:
			nilType = "nil maps"
		case *types.Chan:
			nilType = "nil channels"
		default:
			return
		}
		ReportNodefFG(pass, expr, "should omit nil check; len() for %s is defined as zero", nilType)
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.BinaryExpr)(nil)}, fn)
	return nil, nil
}

func LintSlicing(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		n := node.(*ast.SliceExpr)
		if n.Max != nil {
			return
		}
		s, ok := n.X.(*ast.Ident)
		if !ok || s.Obj == nil {
			return
		}
		call, ok := n.High.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 || call.Ellipsis.IsValid() {
			return
		}
		fun, ok := call.Fun.(*ast.Ident)
		if !ok || fun.Name != "len" {
			return
		}
		if _, ok := pass.TypesInfo.ObjectOf(fun).(*types.Builtin); !ok {
			return
		}
		arg, ok := call.Args[Arg("len.v")].(*ast.Ident)
		if !ok || arg.Obj != s.Obj {
			return
		}
		ReportNodefFG(pass, n, "should omit second index in slice, s[a:len(s)] is identical to s[a:]")
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.SliceExpr)(nil)}, fn)
	return nil, nil
}

func refersTo(pass *analysis.Pass, expr ast.Expr, ident *ast.Ident) bool {
	found := false
	fn := func(node ast.Node) bool {
		ident2, ok := node.(*ast.Ident)
		if !ok {
			return true
		}
		if pass.TypesInfo.ObjectOf(ident) == pass.TypesInfo.ObjectOf(ident2) {
			found = true
			return false
		}
		return true
	}
	ast.Inspect(expr, fn)
	return found
}

func LintLoopAppend(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		loop := node.(*ast.RangeStmt)
		if !IsBlank(loop.Key) {
			return
		}
		val, ok := loop.Value.(*ast.Ident)
		if !ok {
			return
		}
		if len(loop.Body.List) != 1 {
			return
		}
		stmt, ok := loop.Body.List[0].(*ast.AssignStmt)
		if !ok {
			return
		}
		if stmt.Tok != token.ASSIGN || len(stmt.Lhs) != 1 || len(stmt.Rhs) != 1 {
			return
		}
		if refersTo(pass, stmt.Lhs[0], val) {
			return
		}
		call, ok := stmt.Rhs[0].(*ast.CallExpr)
		if !ok {
			return
		}
		if len(call.Args) != 2 || call.Ellipsis.IsValid() {
			return
		}
		fun, ok := call.Fun.(*ast.Ident)
		if !ok {
			return
		}
		obj := pass.TypesInfo.ObjectOf(fun)
		fn, ok := obj.(*types.Builtin)
		if !ok || fn.Name() != "append" {
			return
		}

		src := pass.TypesInfo.TypeOf(loop.X)
		dst := pass.TypesInfo.TypeOf(call.Args[Arg("append.slice")])
		// TODO(dominikh) remove nil check once Go issue #15173 has
		// been fixed
		if src == nil {
			return
		}
		if !types.Identical(src, dst) {
			return
		}

		if Render(pass, stmt.Lhs[0]) != Render(pass, call.Args[Arg("append.slice")]) {
			return
		}

		el, ok := call.Args[Arg("append.elems")].(*ast.Ident)
		if !ok {
			return
		}
		if pass.TypesInfo.ObjectOf(val) != pass.TypesInfo.ObjectOf(el) {
			return
		}
		ReportNodefFG(pass, loop, "should replace loop with %s = append(%s, %s...)",
			Render(pass, stmt.Lhs[0]), Render(pass, call.Args[Arg("append.slice")]), Render(pass, loop.X))
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.RangeStmt)(nil)}, fn)
	return nil, nil
}

func LintTimeSince(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if !IsCallToAST(pass, sel.X, "time.Now") {
			return
		}
		if sel.Sel.Name != "Sub" {
			return
		}
		ReportNodefFG(pass, call, "should use time.Since instead of time.Now().Sub")
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.CallExpr)(nil)}, fn)
	return nil, nil
}

func LintTimeUntil(pass *analysis.Pass) (interface{}, error) {
	if !IsGoVersion(pass, 8) {
		return nil, nil
	}
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		if !IsCallToAST(pass, call, "(time.Time).Sub") {
			return
		}
		if !IsCallToAST(pass, call.Args[Arg("(time.Time).Sub.u")], "time.Now") {
			return
		}
		ReportNodefFG(pass, call, "should use time.Until instead of t.Sub(time.Now())")
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.CallExpr)(nil)}, fn)
	return nil, nil
}

func LintUnnecessaryBlank(pass *analysis.Pass) (interface{}, error) {
	fn1 := func(node ast.Node) {
		assign := node.(*ast.AssignStmt)
		if len(assign.Lhs) != 2 || len(assign.Rhs) != 1 {
			return
		}
		if !IsBlank(assign.Lhs[1]) {
			return
		}
		switch rhs := assign.Rhs[0].(type) {
		case *ast.IndexExpr:
			// The type-checker should make sure that it's a map, but
			// let's be safe.
			if _, ok := pass.TypesInfo.TypeOf(rhs.X).Underlying().(*types.Map); !ok {
				return
			}
		case *ast.UnaryExpr:
			if rhs.Op != token.ARROW {
				return
			}
		default:
			return
		}
		cp := *assign
		cp.Lhs = cp.Lhs[0:1]
		ReportNodefFG(pass, assign, "should write %s instead of %s", Render(pass, &cp), Render(pass, assign))
	}

	fn2 := func(node ast.Node) {
		stmt := node.(*ast.AssignStmt)
		if len(stmt.Lhs) != len(stmt.Rhs) {
			return
		}
		for i, lh := range stmt.Lhs {
			rh := stmt.Rhs[i]
			if !IsBlank(lh) {
				continue
			}
			expr, ok := rh.(*ast.UnaryExpr)
			if !ok {
				continue
			}
			if expr.Op != token.ARROW {
				continue
			}
			ReportNodefFG(pass, lh, "'_ = <-ch' can be simplified to '<-ch'")
		}
	}

	fn3 := func(node ast.Node) {
		rs := node.(*ast.RangeStmt)

		// for x, _
		if !IsBlank(rs.Key) && IsBlank(rs.Value) {
			ReportNodefFG(pass, rs.Value, "should omit value from range; this loop is equivalent to `for %s %s range ...`", Render(pass, rs.Key), rs.Tok)
		}
		// for _, _ || for _
		if IsBlank(rs.Key) && (IsBlank(rs.Value) || rs.Value == nil) {
			ReportNodefFG(pass, rs.Key, "should omit values from range; this loop is equivalent to `for range ...`")
		}
	}

	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.AssignStmt)(nil)}, fn1)
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.AssignStmt)(nil)}, fn2)
	if IsGoVersion(pass, 4) {
		pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.RangeStmt)(nil)}, fn3)
	}
	return nil, nil
}

func LintSimplerStructConversion(pass *analysis.Pass) (interface{}, error) {
	var skip ast.Node
	fn := func(node ast.Node) {
		// Do not suggest type conversion between pointers
		if unary, ok := node.(*ast.UnaryExpr); ok && unary.Op == token.AND {
			if lit, ok := unary.X.(*ast.CompositeLit); ok {
				skip = lit
			}
			return
		}

		if node == skip {
			return
		}

		lit, ok := node.(*ast.CompositeLit)
		if !ok {
			return
		}
		typ1, _ := pass.TypesInfo.TypeOf(lit.Type).(*types.Named)
		if typ1 == nil {
			return
		}
		s1, ok := typ1.Underlying().(*types.Struct)
		if !ok {
			return
		}

		var typ2 *types.Named
		var ident *ast.Ident
		getSelType := func(expr ast.Expr) (types.Type, *ast.Ident, bool) {
			sel, ok := expr.(*ast.SelectorExpr)
			if !ok {
				return nil, nil, false
			}
			ident, ok := sel.X.(*ast.Ident)
			if !ok {
				return nil, nil, false
			}
			typ := pass.TypesInfo.TypeOf(sel.X)
			return typ, ident, typ != nil
		}
		if len(lit.Elts) == 0 {
			return
		}
		if s1.NumFields() != len(lit.Elts) {
			return
		}
		for i, elt := range lit.Elts {
			var t types.Type
			var id *ast.Ident
			var ok bool
			switch elt := elt.(type) {
			case *ast.SelectorExpr:
				t, id, ok = getSelType(elt)
				if !ok {
					return
				}
				if i >= s1.NumFields() || s1.Field(i).Name() != elt.Sel.Name {
					return
				}
			case *ast.KeyValueExpr:
				var sel *ast.SelectorExpr
				sel, ok = elt.Value.(*ast.SelectorExpr)
				if !ok {
					return
				}

				if elt.Key.(*ast.Ident).Name != sel.Sel.Name {
					return
				}
				t, id, ok = getSelType(elt.Value)
			}
			if !ok {
				return
			}
			// All fields must be initialized from the same object
			if ident != nil && ident.Obj != id.Obj {
				return
			}
			typ2, _ = t.(*types.Named)
			if typ2 == nil {
				return
			}
			ident = id
		}

		if typ2 == nil {
			return
		}

		if typ1.Obj().Pkg() != typ2.Obj().Pkg() {
			// Do not suggest type conversions between different
			// packages. Types in different packages might only match
			// by coincidence. Furthermore, if the dependency ever
			// adds more fields to its type, it could break the code
			// that relies on the type conversion to work.
			return
		}

		s2, ok := typ2.Underlying().(*types.Struct)
		if !ok {
			return
		}
		if typ1 == typ2 {
			return
		}
		if IsGoVersion(pass, 8) {
			if !types.IdenticalIgnoreTags(s1, s2) {
				return
			}
		} else {
			if !types.Identical(s1, s2) {
				return
			}
		}
		ReportNodefFG(pass, node, "should convert %s (type %s) to %s instead of using struct literal",
			ident.Name, typ2.Obj().Name(), typ1.Obj().Name())
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.UnaryExpr)(nil), (*ast.CompositeLit)(nil)}, fn)
	return nil, nil
}

func LintTrim(pass *analysis.Pass) (interface{}, error) {
	sameNonDynamic := func(node1, node2 ast.Node) bool {
		if reflect.TypeOf(node1) != reflect.TypeOf(node2) {
			return false
		}

		switch node1 := node1.(type) {
		case *ast.Ident:
			return node1.Obj == node2.(*ast.Ident).Obj
		case *ast.SelectorExpr:
			return Render(pass, node1) == Render(pass, node2)
		case *ast.IndexExpr:
			return Render(pass, node1) == Render(pass, node2)
		}
		return false
	}

	isLenOnIdent := func(fn ast.Expr, ident ast.Expr) bool {
		call, ok := fn.(*ast.CallExpr)
		if !ok {
			return false
		}
		if fn, ok := call.Fun.(*ast.Ident); !ok || fn.Name != "len" {
			return false
		}
		if len(call.Args) != 1 {
			return false
		}
		return sameNonDynamic(call.Args[Arg("len.v")], ident)
	}

	fn := func(node ast.Node) {
		var pkg string
		var fun string

		ifstmt := node.(*ast.IfStmt)
		if ifstmt.Init != nil {
			return
		}
		if ifstmt.Else != nil {
			return
		}
		if len(ifstmt.Body.List) != 1 {
			return
		}
		condCall, ok := ifstmt.Cond.(*ast.CallExpr)
		if !ok {
			return
		}
		switch {
		case IsCallToAST(pass, condCall, "strings.HasPrefix"):
			pkg = "strings"
			fun = "HasPrefix"
		case IsCallToAST(pass, condCall, "strings.HasSuffix"):
			pkg = "strings"
			fun = "HasSuffix"
		case IsCallToAST(pass, condCall, "strings.Contains"):
			pkg = "strings"
			fun = "Contains"
		case IsCallToAST(pass, condCall, "bytes.HasPrefix"):
			pkg = "bytes"
			fun = "HasPrefix"
		case IsCallToAST(pass, condCall, "bytes.HasSuffix"):
			pkg = "bytes"
			fun = "HasSuffix"
		case IsCallToAST(pass, condCall, "bytes.Contains"):
			pkg = "bytes"
			fun = "Contains"
		default:
			return
		}

		assign, ok := ifstmt.Body.List[0].(*ast.AssignStmt)
		if !ok {
			return
		}
		if assign.Tok != token.ASSIGN {
			return
		}
		if len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
			return
		}
		if !sameNonDynamic(condCall.Args[0], assign.Lhs[0]) {
			return
		}

		switch rhs := assign.Rhs[0].(type) {
		case *ast.CallExpr:
			if len(rhs.Args) < 2 || !sameNonDynamic(condCall.Args[0], rhs.Args[0]) || !sameNonDynamic(condCall.Args[1], rhs.Args[1]) {
				return
			}
			if IsCallToAST(pass, condCall, "strings.HasPrefix") && IsCallToAST(pass, rhs, "strings.TrimPrefix") ||
				IsCallToAST(pass, condCall, "strings.HasSuffix") && IsCallToAST(pass, rhs, "strings.TrimSuffix") ||
				IsCallToAST(pass, condCall, "strings.Contains") && IsCallToAST(pass, rhs, "strings.Replace") ||
				IsCallToAST(pass, condCall, "bytes.HasPrefix") && IsCallToAST(pass, rhs, "bytes.TrimPrefix") ||
				IsCallToAST(pass, condCall, "bytes.HasSuffix") && IsCallToAST(pass, rhs, "bytes.TrimSuffix") ||
				IsCallToAST(pass, condCall, "bytes.Contains") && IsCallToAST(pass, rhs, "bytes.Replace") {
				ReportNodefFG(pass, ifstmt, "should replace this if statement with an unconditional %s", CallNameAST(pass, rhs))
			}
			return
		case *ast.SliceExpr:
			slice := rhs
			if !ok {
				return
			}
			if slice.Slice3 {
				return
			}
			if !sameNonDynamic(slice.X, condCall.Args[0]) {
				return
			}
			var index ast.Expr
			switch fun {
			case "HasPrefix":
				// TODO(dh) We could detect a High that is len(s), but another
				// rule will already flag that, anyway.
				if slice.High != nil {
					return
				}
				index = slice.Low
			case "HasSuffix":
				if slice.Low != nil {
					n, ok := ExprToInt(pass, slice.Low)
					if !ok || n != 0 {
						return
					}
				}
				index = slice.High
			}

			switch index := index.(type) {
			case *ast.CallExpr:
				if fun != "HasPrefix" {
					return
				}
				if fn, ok := index.Fun.(*ast.Ident); !ok || fn.Name != "len" {
					return
				}
				if len(index.Args) != 1 {
					return
				}
				id3 := index.Args[Arg("len.v")]
				switch oid3 := condCall.Args[1].(type) {
				case *ast.BasicLit:
					if pkg != "strings" {
						return
					}
					lit, ok := id3.(*ast.BasicLit)
					if !ok {
						return
					}
					s1, ok1 := ExprToString(pass, lit)
					s2, ok2 := ExprToString(pass, condCall.Args[1])
					if !ok1 || !ok2 || s1 != s2 {
						return
					}
				default:
					if !sameNonDynamic(id3, oid3) {
						return
					}
				}
			case *ast.BasicLit, *ast.Ident:
				if fun != "HasPrefix" {
					return
				}
				if pkg != "strings" {
					return
				}
				string, ok1 := ExprToString(pass, condCall.Args[1])
				int, ok2 := ExprToInt(pass, slice.Low)
				if !ok1 || !ok2 || int != int64(len(string)) {
					return
				}
			case *ast.BinaryExpr:
				if fun != "HasSuffix" {
					return
				}
				if index.Op != token.SUB {
					return
				}
				if !isLenOnIdent(index.X, condCall.Args[0]) ||
					!isLenOnIdent(index.Y, condCall.Args[1]) {
					return
				}
			default:
				return
			}

			var replacement string
			switch fun {
			case "HasPrefix":
				replacement = "TrimPrefix"
			case "HasSuffix":
				replacement = "TrimSuffix"
			}
			ReportNodefFG(pass, ifstmt, "should replace this if statement with an unconditional %s.%s", pkg, replacement)
		}
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.IfStmt)(nil)}, fn)
	return nil, nil
}

func LintLoopSlide(pass *analysis.Pass) (interface{}, error) {
	// TODO(dh): detect bs[i+offset] in addition to bs[offset+i]
	// TODO(dh): consider merging this function with LintLoopCopy
	// TODO(dh): detect length that is an expression, not a variable name
	// TODO(dh): support sliding to a different offset than the beginning of the slice

	fn := func(node ast.Node) {
		/*
			for i := 0; i < n; i++ {
				bs[i] = bs[offset+i]
			}

						↓

			copy(bs[:n], bs[offset:offset+n])
		*/

		loop := node.(*ast.ForStmt)
		if len(loop.Body.List) != 1 || loop.Init == nil || loop.Cond == nil || loop.Post == nil {
			return
		}
		assign, ok := loop.Init.(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 || !IsZero(assign.Rhs[0]) {
			return
		}
		initvar, ok := assign.Lhs[0].(*ast.Ident)
		if !ok {
			return
		}
		post, ok := loop.Post.(*ast.IncDecStmt)
		if !ok || post.Tok != token.INC {
			return
		}
		postvar, ok := post.X.(*ast.Ident)
		if !ok || pass.TypesInfo.ObjectOf(postvar) != pass.TypesInfo.ObjectOf(initvar) {
			return
		}
		bin, ok := loop.Cond.(*ast.BinaryExpr)
		if !ok || bin.Op != token.LSS {
			return
		}
		binx, ok := bin.X.(*ast.Ident)
		if !ok || pass.TypesInfo.ObjectOf(binx) != pass.TypesInfo.ObjectOf(initvar) {
			return
		}
		biny, ok := bin.Y.(*ast.Ident)
		if !ok {
			return
		}

		assign, ok = loop.Body.List[0].(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 1 || len(assign.Rhs) != 1 || assign.Tok != token.ASSIGN {
			return
		}
		lhs, ok := assign.Lhs[0].(*ast.IndexExpr)
		if !ok {
			return
		}
		rhs, ok := assign.Rhs[0].(*ast.IndexExpr)
		if !ok {
			return
		}

		bs1, ok := lhs.X.(*ast.Ident)
		if !ok {
			return
		}
		bs2, ok := rhs.X.(*ast.Ident)
		if !ok {
			return
		}
		obj1 := pass.TypesInfo.ObjectOf(bs1)
		obj2 := pass.TypesInfo.ObjectOf(bs2)
		if obj1 != obj2 {
			return
		}
		if _, ok := obj1.Type().Underlying().(*types.Slice); !ok {
			return
		}

		index1, ok := lhs.Index.(*ast.Ident)
		if !ok || pass.TypesInfo.ObjectOf(index1) != pass.TypesInfo.ObjectOf(initvar) {
			return
		}
		index2, ok := rhs.Index.(*ast.BinaryExpr)
		if !ok || index2.Op != token.ADD {
			return
		}
		add1, ok := index2.X.(*ast.Ident)
		if !ok {
			return
		}
		add2, ok := index2.Y.(*ast.Ident)
		if !ok || pass.TypesInfo.ObjectOf(add2) != pass.TypesInfo.ObjectOf(initvar) {
			return
		}

		ReportNodefFG(pass, loop, "should use copy(%s[:%s], %s[%s:]) instead", Render(pass, bs1), Render(pass, biny), Render(pass, bs1), Render(pass, add1))
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.ForStmt)(nil)}, fn)
	return nil, nil
}

func LintMakeLenCap(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		if fn, ok := call.Fun.(*ast.Ident); !ok || fn.Name != "make" {
			// FIXME check whether make is indeed the built-in function
			return
		}
		switch len(call.Args) {
		case 2:
			// make(T, len)
			if _, ok := pass.TypesInfo.TypeOf(call.Args[Arg("make.t")]).Underlying().(*types.Slice); ok {
				break
			}
			if IsZero(call.Args[Arg("make.size[0]")]) {
				ReportNodefFG(pass, call.Args[Arg("make.size[0]")], "should use make(%s) instead", Render(pass, call.Args[Arg("make.t")]))
			}
		case 3:
			// make(T, len, cap)
			if Render(pass, call.Args[Arg("make.size[0]")]) == Render(pass, call.Args[Arg("make.size[1]")]) {
				ReportNodefFG(pass, call.Args[Arg("make.size[0]")],
					"should use make(%s, %s) instead",
					Render(pass, call.Args[Arg("make.t")]), Render(pass, call.Args[Arg("make.size[0]")]))
			}
		}
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.CallExpr)(nil)}, fn)
	return nil, nil
}

func LintAssertNotNil(pass *analysis.Pass) (interface{}, error) {
	isNilCheck := func(ident *ast.Ident, expr ast.Expr) bool {
		xbinop, ok := expr.(*ast.BinaryExpr)
		if !ok || xbinop.Op != token.NEQ {
			return false
		}
		xident, ok := xbinop.X.(*ast.Ident)
		if !ok || xident.Obj != ident.Obj {
			return false
		}
		if !IsNil(pass, xbinop.Y) {
			return false
		}
		return true
	}
	isOKCheck := func(ident *ast.Ident, expr ast.Expr) bool {
		yident, ok := expr.(*ast.Ident)
		if !ok || yident.Obj != ident.Obj {
			return false
		}
		return true
	}
	fn1 := func(node ast.Node) {
		ifstmt := node.(*ast.IfStmt)
		assign, ok := ifstmt.Init.(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 2 || len(assign.Rhs) != 1 || !IsBlank(assign.Lhs[0]) {
			return
		}
		assert, ok := assign.Rhs[0].(*ast.TypeAssertExpr)
		if !ok {
			return
		}
		binop, ok := ifstmt.Cond.(*ast.BinaryExpr)
		if !ok || binop.Op != token.LAND {
			return
		}
		assertIdent, ok := assert.X.(*ast.Ident)
		if !ok {
			return
		}
		assignIdent, ok := assign.Lhs[1].(*ast.Ident)
		if !ok {
			return
		}
		if !(isNilCheck(assertIdent, binop.X) && isOKCheck(assignIdent, binop.Y)) &&
			!(isNilCheck(assertIdent, binop.Y) && isOKCheck(assignIdent, binop.X)) {
			return
		}
		ReportNodefFG(pass, ifstmt, "when %s is true, %s can't be nil", Render(pass, assignIdent), Render(pass, assertIdent))
	}
	fn2 := func(node ast.Node) {
		// Check that outer ifstmt is an 'if x != nil {}'
		ifstmt := node.(*ast.IfStmt)
		if ifstmt.Init != nil {
			return
		}
		if ifstmt.Else != nil {
			return
		}
		if len(ifstmt.Body.List) != 1 {
			return
		}
		binop, ok := ifstmt.Cond.(*ast.BinaryExpr)
		if !ok {
			return
		}
		if binop.Op != token.NEQ {
			return
		}
		lhs, ok := binop.X.(*ast.Ident)
		if !ok {
			return
		}
		if !IsNil(pass, binop.Y) {
			return
		}

		// Check that inner ifstmt is an `if _, ok := x.(T); ok {}`
		ifstmt, ok = ifstmt.Body.List[0].(*ast.IfStmt)
		if !ok {
			return
		}
		assign, ok := ifstmt.Init.(*ast.AssignStmt)
		if !ok || len(assign.Lhs) != 2 || len(assign.Rhs) != 1 || !IsBlank(assign.Lhs[0]) {
			return
		}
		assert, ok := assign.Rhs[0].(*ast.TypeAssertExpr)
		if !ok {
			return
		}
		assertIdent, ok := assert.X.(*ast.Ident)
		if !ok {
			return
		}
		if lhs.Obj != assertIdent.Obj {
			return
		}
		assignIdent, ok := assign.Lhs[1].(*ast.Ident)
		if !ok {
			return
		}
		if !isOKCheck(assignIdent, ifstmt.Cond) {
			return
		}
		ReportNodefFG(pass, ifstmt, "when %s is true, %s can't be nil", Render(pass, assignIdent), Render(pass, assertIdent))
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.IfStmt)(nil)}, fn1)
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.IfStmt)(nil)}, fn2)
	return nil, nil
}

func LintDeclareAssign(pass *analysis.Pass) (interface{}, error) {
	hasMultipleAssignments := func(root ast.Node, ident *ast.Ident) bool {
		num := 0
		ast.Inspect(root, func(node ast.Node) bool {
			if num >= 2 {
				return false
			}
			assign, ok := node.(*ast.AssignStmt)
			if !ok {
				return true
			}
			for _, lhs := range assign.Lhs {
				if oident, ok := lhs.(*ast.Ident); ok {
					if oident.Obj == ident.Obj {
						num++
					}
				}
			}

			return true
		})
		return num >= 2
	}
	fn := func(node ast.Node) {
		block := node.(*ast.BlockStmt)
		if len(block.List) < 2 {
			return
		}
		for i, stmt := range block.List[:len(block.List)-1] {
			_ = i
			decl, ok := stmt.(*ast.DeclStmt)
			if !ok {
				continue
			}
			gdecl, ok := decl.Decl.(*ast.GenDecl)
			if !ok || gdecl.Tok != token.VAR || len(gdecl.Specs) != 1 {
				continue
			}
			vspec, ok := gdecl.Specs[0].(*ast.ValueSpec)
			if !ok || len(vspec.Names) != 1 || len(vspec.Values) != 0 {
				continue
			}

			assign, ok := block.List[i+1].(*ast.AssignStmt)
			if !ok || assign.Tok != token.ASSIGN {
				continue
			}
			if len(assign.Lhs) != 1 || len(assign.Rhs) != 1 {
				continue
			}
			ident, ok := assign.Lhs[0].(*ast.Ident)
			if !ok {
				continue
			}
			if vspec.Names[0].Obj != ident.Obj {
				continue
			}

			if refersTo(pass, assign.Rhs[0], ident) {
				continue
			}
			if hasMultipleAssignments(block, ident) {
				continue
			}

			ReportNodefFG(pass, decl, "should merge variable declaration with assignment on next line")
		}
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.BlockStmt)(nil)}, fn)
	return nil, nil
}

func LintRedundantBreak(pass *analysis.Pass) (interface{}, error) {
	fn1 := func(node ast.Node) {
		clause := node.(*ast.CaseClause)
		if len(clause.Body) < 2 {
			return
		}
		branch, ok := clause.Body[len(clause.Body)-1].(*ast.BranchStmt)
		if !ok || branch.Tok != token.BREAK || branch.Label != nil {
			return
		}
		ReportNodefFG(pass, branch, "redundant break statement")
	}
	fn2 := func(node ast.Node) {
		var ret *ast.FieldList
		var body *ast.BlockStmt
		switch x := node.(type) {
		case *ast.FuncDecl:
			ret = x.Type.Results
			body = x.Body
		case *ast.FuncLit:
			ret = x.Type.Results
			body = x.Body
		default:
			panic(fmt.Sprintf("unreachable: %T", node))
		}
		// if the func has results, a return can't be redundant.
		// similarly, if there are no statements, there can be
		// no return.
		if ret != nil || body == nil || len(body.List) < 1 {
			return
		}
		rst, ok := body.List[len(body.List)-1].(*ast.ReturnStmt)
		if !ok {
			return
		}
		// we don't need to check rst.Results as we already
		// checked x.Type.Results to be nil.
		ReportNodefFG(pass, rst, "redundant return statement")
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.CaseClause)(nil)}, fn1)
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.FuncDecl)(nil), (*ast.FuncLit)(nil)}, fn2)
	return nil, nil
}

func isStringer(T types.Type, msCache *typeutil.MethodSetCache) bool {
	ms := msCache.MethodSet(T)
	sel := ms.Lookup(nil, "String")
	if sel == nil {
		return false
	}
	fn, ok := sel.Obj().(*types.Func)
	if !ok {
		// should be unreachable
		return false
	}
	sig := fn.Type().(*types.Signature)
	if sig.Params().Len() != 0 {
		return false
	}
	if sig.Results().Len() != 1 {
		return false
	}
	if !IsType(sig.Results().At(0).Type(), "string") {
		return false
	}
	return true
}

func LintRedundantSprintf(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		call := node.(*ast.CallExpr)
		if !IsCallToAST(pass, call, "fmt.Sprintf") {
			return
		}
		if len(call.Args) != 2 {
			return
		}
		if s, ok := ExprToString(pass, call.Args[Arg("fmt.Sprintf.format")]); !ok || s != "%s" {
			return
		}
		arg := call.Args[Arg("fmt.Sprintf.a[0]")]
		typ := pass.TypesInfo.TypeOf(arg)

		ssapkg := pass.ResultOf[buildssa.Analyzer].(*buildssa.SSA).Pkg
		if isStringer(typ, &ssapkg.Prog.MethodSets) {
			ReportNodef(pass, call, "should use String() instead of fmt.Sprintf")
			return
		}

		if typ.Underlying() == types.Universe.Lookup("string").Type() {
			if typ == types.Universe.Lookup("string").Type() {
				ReportNodefFG(pass, call, "the argument is already a string, there's no need to use fmt.Sprintf")
			} else {
				ReportNodefFG(pass, call, "the argument's underlying type is a string, should use a simple conversion instead of fmt.Sprintf")
			}
		}
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.CallExpr)(nil)}, fn)
	return nil, nil
}

func LintErrorsNewSprintf(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		if !IsCallToAST(pass, node, "errors.New") {
			return
		}
		call := node.(*ast.CallExpr)
		if !IsCallToAST(pass, call.Args[Arg("errors.New.text")], "fmt.Sprintf") {
			return
		}
		ReportNodefFG(pass, node, "should use fmt.Errorf(...) instead of errors.New(fmt.Sprintf(...))")
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.CallExpr)(nil)}, fn)
	return nil, nil
}

func LintRangeStringRunes(pass *analysis.Pass) (interface{}, error) {
	return sharedcheck.CheckRangeStringRunes(pass)
}

func LintNilCheckAroundRange(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		ifstmt := node.(*ast.IfStmt)
		cond, ok := ifstmt.Cond.(*ast.BinaryExpr)
		if !ok {
			return
		}

		if cond.Op != token.NEQ || !IsNil(pass, cond.Y) || len(ifstmt.Body.List) != 1 {
			return
		}

		loop, ok := ifstmt.Body.List[0].(*ast.RangeStmt)
		if !ok {
			return
		}
		ifXIdent, ok := cond.X.(*ast.Ident)
		if !ok {
			return
		}
		rangeXIdent, ok := loop.X.(*ast.Ident)
		if !ok {
			return
		}
		if ifXIdent.Obj != rangeXIdent.Obj {
			return
		}
		switch pass.TypesInfo.TypeOf(rangeXIdent).(type) {
		case *types.Slice, *types.Map:
			ReportNodefFG(pass, node, "unnecessary nil check around range")
		}
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.IfStmt)(nil)}, fn)
	return nil, nil
}

func isPermissibleSort(pass *analysis.Pass, node ast.Node) bool {
	call := node.(*ast.CallExpr)
	typeconv, ok := call.Args[0].(*ast.CallExpr)
	if !ok {
		return true
	}

	sel, ok := typeconv.Fun.(*ast.SelectorExpr)
	if !ok {
		return true
	}
	name := SelectorName(pass, sel)
	switch name {
	case "sort.IntSlice", "sort.Float64Slice", "sort.StringSlice":
	default:
		return true
	}

	return false
}

func LintSortHelpers(pass *analysis.Pass) (interface{}, error) {
	type Error struct {
		node ast.Node
		msg  string
	}
	var allErrors []Error
	fn := func(node ast.Node) {
		var body *ast.BlockStmt
		switch node := node.(type) {
		case *ast.FuncLit:
			body = node.Body
		case *ast.FuncDecl:
			body = node.Body
		default:
			panic(fmt.Sprintf("unreachable: %T", node))
		}
		if body == nil {
			return
		}

		var errors []Error
		permissible := false
		fnSorts := func(node ast.Node) bool {
			if permissible {
				return false
			}
			if !IsCallToAST(pass, node, "sort.Sort") {
				return true
			}
			if isPermissibleSort(pass, node) {
				permissible = true
				return false
			}
			call := node.(*ast.CallExpr)
			typeconv := call.Args[Arg("sort.Sort.data")].(*ast.CallExpr)
			sel := typeconv.Fun.(*ast.SelectorExpr)
			name := SelectorName(pass, sel)

			switch name {
			case "sort.IntSlice":
				errors = append(errors, Error{node, "should use sort.Ints(...) instead of sort.Sort(sort.IntSlice(...))"})
			case "sort.Float64Slice":
				errors = append(errors, Error{node, "should use sort.Float64s(...) instead of sort.Sort(sort.Float64Slice(...))"})
			case "sort.StringSlice":
				errors = append(errors, Error{node, "should use sort.Strings(...) instead of sort.Sort(sort.StringSlice(...))"})
			}
			return true
		}
		ast.Inspect(body, fnSorts)

		if permissible {
			return
		}
		allErrors = append(allErrors, errors...)
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.FuncLit)(nil), (*ast.FuncDecl)(nil)}, fn)
	sort.Slice(allErrors, func(i, j int) bool {
		return allErrors[i].node.Pos() < allErrors[j].node.Pos()
	})
	var prev token.Pos
	for _, err := range allErrors {
		if err.node.Pos() == prev {
			continue
		}
		prev = err.node.Pos()
		ReportNodefFG(pass, err.node, "%s", err.msg)
	}
	return nil, nil
}

func LintGuardedDelete(pass *analysis.Pass) (interface{}, error) {
	isCommaOkMapIndex := func(stmt ast.Stmt) (b *ast.Ident, m ast.Expr, key ast.Expr, ok bool) {
		// Has to be of the form `_, <b:*ast.Ident> = <m:*types.Map>[<key>]

		assign, ok := stmt.(*ast.AssignStmt)
		if !ok {
			return nil, nil, nil, false
		}
		if len(assign.Lhs) != 2 || len(assign.Rhs) != 1 {
			return nil, nil, nil, false
		}
		if !IsBlank(assign.Lhs[0]) {
			return nil, nil, nil, false
		}
		ident, ok := assign.Lhs[1].(*ast.Ident)
		if !ok {
			return nil, nil, nil, false
		}
		index, ok := assign.Rhs[0].(*ast.IndexExpr)
		if !ok {
			return nil, nil, nil, false
		}
		if _, ok := pass.TypesInfo.TypeOf(index.X).(*types.Map); !ok {
			return nil, nil, nil, false
		}
		key = index.Index
		return ident, index.X, key, true
	}
	fn := func(node ast.Node) {
		stmt := node.(*ast.IfStmt)
		if len(stmt.Body.List) != 1 {
			return
		}
		if stmt.Else != nil {
			return
		}
		expr, ok := stmt.Body.List[0].(*ast.ExprStmt)
		if !ok {
			return
		}
		call, ok := expr.X.(*ast.CallExpr)
		if !ok {
			return
		}
		if !IsCallToAST(pass, call, "delete") {
			return
		}
		b, m, key, ok := isCommaOkMapIndex(stmt.Init)
		if !ok {
			return
		}
		if cond, ok := stmt.Cond.(*ast.Ident); !ok || pass.TypesInfo.ObjectOf(cond) != pass.TypesInfo.ObjectOf(b) {
			return
		}
		if Render(pass, call.Args[0]) != Render(pass, m) || Render(pass, call.Args[1]) != Render(pass, key) {
			return
		}
		ReportNodefFG(pass, stmt, "unnecessary guard around call to delete")
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.IfStmt)(nil)}, fn)
	return nil, nil
}

func LintSimplifyTypeSwitch(pass *analysis.Pass) (interface{}, error) {
	fn := func(node ast.Node) {
		stmt := node.(*ast.TypeSwitchStmt)
		if stmt.Init != nil {
			// bailing out for now, can't anticipate how type switches with initializers are being used
			return
		}
		expr, ok := stmt.Assign.(*ast.ExprStmt)
		if !ok {
			// the user is in fact assigning the result
			return
		}
		assert := expr.X.(*ast.TypeAssertExpr)
		ident, ok := assert.X.(*ast.Ident)
		if !ok {
			return
		}
		x := pass.TypesInfo.ObjectOf(ident)
		var allOffenders []ast.Node
		for _, clause := range stmt.Body.List {
			clause := clause.(*ast.CaseClause)
			if len(clause.List) != 1 {
				continue
			}
			hasUnrelatedAssertion := false
			var offenders []ast.Node
			ast.Inspect(clause, func(node ast.Node) bool {
				assert2, ok := node.(*ast.TypeAssertExpr)
				if !ok {
					return true
				}
				ident, ok := assert2.X.(*ast.Ident)
				if !ok {
					hasUnrelatedAssertion = true
					return false
				}
				if pass.TypesInfo.ObjectOf(ident) != x {
					hasUnrelatedAssertion = true
					return false
				}

				if !types.Identical(pass.TypesInfo.TypeOf(clause.List[0]), pass.TypesInfo.TypeOf(assert2.Type)) {
					hasUnrelatedAssertion = true
					return false
				}
				offenders = append(offenders, assert2)
				return true
			})
			if !hasUnrelatedAssertion {
				// don't flag cases that have other type assertions
				// unrelated to the one in the case clause. often
				// times, this is done for symmetry, when two
				// different values have to be asserted to the same
				// type.
				allOffenders = append(allOffenders, offenders...)
			}
		}
		if len(allOffenders) != 0 {
			at := ""
			for _, offender := range allOffenders {
				pos := lint.DisplayPosition(pass.Fset, offender.Pos())
				at += "\n\t" + pos.String()
			}
			ReportNodefFG(pass, expr, "assigning the result of this type assertion to a variable (switch %s := %s.(type)) could eliminate the following type assertions:%s", Render(pass, ident), Render(pass, ident), at)
		}
	}
	pass.ResultOf[inspect.Analyzer].(*inspector.Inspector).Preorder([]ast.Node{(*ast.TypeSwitchStmt)(nil)}, fn)
	return nil, nil
}
