package ruleguard

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"path/filepath"

	"github.com/quasilyte/go-ruleguard/internal/xtypes"
	"github.com/quasilyte/go-ruleguard/ruleguard/quasigo"
	"github.com/quasilyte/go-ruleguard/ruleguard/textmatch"
	"github.com/quasilyte/go-ruleguard/ruleguard/typematch"
	"github.com/quasilyte/gogrep"
	"github.com/quasilyte/gogrep/nodetag"
)

const filterSuccess = matchFilterResult("")

func filterFailure(reason string) matchFilterResult {
	return matchFilterResult(reason)
}

func exprListFilterApply(src string, list gogrep.ExprSlice, fn func(ast.Expr) bool) matchFilterResult {
	for i := 0; i < list.Len(); i++ {
		if !fn(list.At(i).(ast.Expr)) {
			return filterFailure(src)
		}
	}
	return filterSuccess
}

func makeNotFilter(src string, x matchFilter) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if x.fn(params).Matched() {
			return matchFilterResult(src)
		}
		return ""
	}
}

func makeAndFilter(lhs, rhs matchFilter) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if lhsResult := lhs.fn(params); !lhsResult.Matched() {
			return lhsResult
		}
		return rhs.fn(params)
	}
}

func makeOrFilter(lhs, rhs matchFilter) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if lhsResult := lhs.fn(params); lhsResult.Matched() {
			return filterSuccess
		}
		return rhs.fn(params)
	}
}

func makeDeadcodeFilter(src string) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if params.deadcode {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeFileImportsFilter(src, pkgPath string) filterFunc {
	return func(params *filterParams) matchFilterResult {
		_, imported := params.imports[pkgPath]
		if imported {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeFilePkgPathMatchesFilter(src string, re textmatch.Pattern) filterFunc {
	return func(params *filterParams) matchFilterResult {
		pkgPath := params.ctx.Pkg.Path()
		if re.MatchString(pkgPath) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeFileNameMatchesFilter(src string, re textmatch.Pattern) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if re.MatchString(filepath.Base(params.filename)) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makePureFilter(src, varname string) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				return isPure(params.ctx.Types, x)
			})
		}

		n := params.subExpr(varname)
		if isPure(params.ctx.Types, n) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeConstFilter(src, varname string) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				return isConstant(params.ctx.Types, x)
			})
		}

		n := params.subExpr(varname)
		if isConstant(params.ctx.Types, n) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeConstSliceFilter(src, varname string) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				return isConstantSlice(params.ctx.Types, x)
			})
		}

		n := params.subExpr(varname)
		if isConstantSlice(params.ctx.Types, n) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeAddressableFilter(src, varname string) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				return isAddressable(params.ctx.Types, x)
			})
		}

		n := params.subExpr(varname)
		if isAddressable(params.ctx.Types, n) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeCustomVarFilter(src, varname string, fn *quasigo.Func) filterFunc {
	return func(params *filterParams) matchFilterResult {
		// TODO(quasilyte): what if bytecode function panics due to the programming error?
		// We should probably catch the panic here, print trace and return "false"
		// from the filter (or even propagate that panic to let it crash).
		params.varname = varname
		result := quasigo.Call(params.env, fn, params)
		if result.Value().(bool) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeTypeImplementsFilter(src, varname string, iface *types.Interface) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				return xtypes.Implements(params.typeofNode(x), iface)
			})
		}

		typ := params.typeofNode(params.subExpr(varname))
		if xtypes.Implements(typ, iface) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeTypeHasPointersFilter(src, varname string) filterFunc {
	return func(params *filterParams) matchFilterResult {
		typ := params.typeofNode(params.subExpr(varname))
		if typeHasPointers(typ) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeTypeIsIntUintFilter(src, varname string, underlying bool, kind types.BasicKind) filterFunc {
	return func(params *filterParams) matchFilterResult {
		typ := params.typeofNode(params.subExpr(varname))
		if underlying {
			typ = typ.Underlying()
		}
		if basicType, ok := typ.(*types.Basic); ok {
			first := kind
			last := kind + 4
			if basicType.Kind() >= first && basicType.Kind() <= last {
				return filterSuccess
			}
		}
		return filterFailure(src)
	}
}

func makeTypeIsSignedFilter(src, varname string, underlying bool) filterFunc {
	return func(params *filterParams) matchFilterResult {
		typ := params.typeofNode(params.subExpr(varname))
		if underlying {
			typ = typ.Underlying()
		}
		if basicType, ok := typ.(*types.Basic); ok {
			if basicType.Info()&types.IsInteger != 0 && basicType.Info()&types.IsUnsigned == 0 {
				return filterSuccess
			}
		}
		return filterFailure(src)
	}
}

func makeTypeOfKindFilter(src, varname string, underlying bool, kind types.BasicInfo) filterFunc {
	return func(params *filterParams) matchFilterResult {
		typ := params.typeofNode(params.subExpr(varname))
		if underlying {
			typ = typ.Underlying()
		}
		if basicType, ok := typ.(*types.Basic); ok {
			if basicType.Info()&kind != 0 {
				return filterSuccess
			}
		}
		return filterFailure(src)
	}
}

func makeTypeIsFilter(src, varname string, underlying bool, pat *typematch.Pattern) filterFunc {
	if underlying {
		return func(params *filterParams) matchFilterResult {
			if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
				return exprListFilterApply(src, list, func(x ast.Expr) bool {
					return pat.MatchIdentical(params.typeofNode(x).Underlying())
				})
			}
			typ := params.typeofNode(params.subNode(varname)).Underlying()
			if pat.MatchIdentical(typ) {
				return filterSuccess
			}
			return filterFailure(src)
		}
	}

	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				return pat.MatchIdentical(params.typeofNode(x))
			})
		}
		typ := params.typeofNode(params.subNode(varname))
		if pat.MatchIdentical(typ) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeTypeConvertibleToFilter(src, varname string, dstType types.Type) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				return types.ConvertibleTo(params.typeofNode(x), dstType)
			})
		}

		typ := params.typeofNode(params.subExpr(varname))
		if types.ConvertibleTo(typ, dstType) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeTypeAssignableToFilter(src, varname string, dstType types.Type) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				return types.AssignableTo(params.typeofNode(x), dstType)
			})
		}

		typ := params.typeofNode(params.subExpr(varname))
		if types.AssignableTo(typ, dstType) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeLineFilter(src, varname string, op token.Token, rhsVarname string) filterFunc {
	// TODO(quasilyte): add variadic support.
	return func(params *filterParams) matchFilterResult {
		line1 := params.ctx.Fset.Position(params.subNode(varname).Pos()).Line
		line2 := params.ctx.Fset.Position(params.subNode(rhsVarname).Pos()).Line
		lhsValue := constant.MakeInt64(int64(line1))
		rhsValue := constant.MakeInt64(int64(line2))
		if constant.Compare(lhsValue, op, rhsValue) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeGoVersionFilter(src string, op token.Token, version GoVersion) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if params.ctx.GoVersion.IsAny() {
			return filterSuccess
		}
		if versionCompare(params.ctx.GoVersion, op, version) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeLineConstFilter(src, varname string, op token.Token, rhsValue constant.Value) filterFunc {
	// TODO(quasilyte): add variadic support.
	return func(params *filterParams) matchFilterResult {
		n := params.subNode(varname)
		lhsValue := constant.MakeInt64(int64(params.ctx.Fset.Position(n.Pos()).Line))
		if constant.Compare(lhsValue, op, rhsValue) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeTypeSizeConstFilter(src, varname string, op token.Token, rhsValue constant.Value) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				typ := params.typeofNode(x)
				lhsValue := constant.MakeInt64(params.ctx.Sizes.Sizeof(typ))
				return constant.Compare(lhsValue, op, rhsValue)
			})
		}

		typ := params.typeofNode(params.subExpr(varname))
		lhsValue := constant.MakeInt64(params.ctx.Sizes.Sizeof(typ))
		if constant.Compare(lhsValue, op, rhsValue) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeValueIntConstFilter(src, varname string, op token.Token, rhsValue constant.Value) filterFunc {
	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				lhsValue := intValueOf(params.ctx.Types, x)
				return lhsValue != nil && constant.Compare(lhsValue, op, rhsValue)
			})
		}

		lhsValue := intValueOf(params.ctx.Types, params.subExpr(varname))
		if lhsValue == nil {
			return filterFailure(src) // The value is unknown
		}
		if constant.Compare(lhsValue, op, rhsValue) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeValueIntFilter(src, varname string, op token.Token, rhsVarname string) filterFunc {
	// TODO(quasilyte): add variadic support.
	return func(params *filterParams) matchFilterResult {
		lhsValue := intValueOf(params.ctx.Types, params.subExpr(varname))
		if lhsValue == nil {
			return filterFailure(src)
		}
		rhsValue := intValueOf(params.ctx.Types, params.subExpr(rhsVarname))
		if rhsValue == nil {
			return filterFailure(src)
		}
		if constant.Compare(lhsValue, op, rhsValue) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeTextConstFilter(src, varname string, op token.Token, rhsValue constant.Value) filterFunc {
	// TODO(quasilyte): add variadic support.
	return func(params *filterParams) matchFilterResult {
		s := params.nodeText(params.subNode(varname))
		lhsValue := constant.MakeString(string(s))
		if constant.Compare(lhsValue, op, rhsValue) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeTextFilter(src, varname string, op token.Token, rhsVarname string) filterFunc {
	// TODO(quasilyte): add variadic support.
	return func(params *filterParams) matchFilterResult {
		s1 := params.nodeText(params.subNode(varname))
		lhsValue := constant.MakeString(string(s1))
		n, _ := params.match.CapturedByName(rhsVarname)
		s2 := params.nodeText(n)
		rhsValue := constant.MakeString(string(s2))
		if constant.Compare(lhsValue, op, rhsValue) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeTextMatchesFilter(src, varname string, re textmatch.Pattern) filterFunc {
	// TODO(quasilyte): add variadic support.
	return func(params *filterParams) matchFilterResult {
		if re.Match(params.nodeText(params.subNode(varname))) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeRootParentNodeIsFilter(src string, tag nodetag.Value) filterFunc {
	return func(params *filterParams) matchFilterResult {
		parent := params.nodePath.Parent()
		if nodeIs(parent, tag) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeNodeIsFilter(src, varname string, tag nodetag.Value) filterFunc {
	// TODO(quasilyte): add comment nodes support?
	// TODO(quasilyte): add variadic support.
	return func(params *filterParams) matchFilterResult {
		n := params.subNode(varname)
		if nodeIs(n, tag) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func makeObjectIsFilter(src, varname, objectName string) filterFunc {
	var predicate func(types.Object) bool
	switch objectName {
	case "Func":
		predicate = func(x types.Object) bool {
			_, ok := x.(*types.Func)
			return ok
		}
	case "Var":
		predicate = func(x types.Object) bool {
			_, ok := x.(*types.Var)
			return ok
		}
	case "Const":
		predicate = func(x types.Object) bool {
			_, ok := x.(*types.Const)
			return ok
		}
	case "TypeName":
		predicate = func(x types.Object) bool {
			_, ok := x.(*types.TypeName)
			return ok
		}
	case "Label":
		predicate = func(x types.Object) bool {
			_, ok := x.(*types.Label)
			return ok
		}
	case "PkgName":
		predicate = func(x types.Object) bool {
			_, ok := x.(*types.PkgName)
			return ok
		}
	case "Builtin":
		predicate = func(x types.Object) bool {
			_, ok := x.(*types.Builtin)
			return ok
		}
	case "Nil":
		predicate = func(x types.Object) bool {
			_, ok := x.(*types.Nil)
			return ok
		}
	}

	return func(params *filterParams) matchFilterResult {
		if list, ok := params.subNode(varname).(gogrep.ExprSlice); ok {
			return exprListFilterApply(src, list, func(x ast.Expr) bool {
				ident := identOf(x)
				return ident != nil && predicate(params.ctx.Types.ObjectOf(ident))
			})
		}

		ident := identOf(params.subExpr(varname))
		if ident == nil {
			return filterFailure(src)
		}
		object := params.ctx.Types.ObjectOf(ident)
		if predicate(object) {
			return filterSuccess
		}
		return filterFailure(src)
	}
}

func nodeIs(n ast.Node, tag nodetag.Value) bool {
	var matched bool
	switch tag {
	case nodetag.Expr:
		_, matched = n.(ast.Expr)
	case nodetag.Stmt:
		_, matched = n.(ast.Stmt)
	case nodetag.Node:
		matched = true
	default:
		matched = (tag == nodetag.FromNode(n))
	}
	return matched
}

func typeHasPointers(typ types.Type) bool {
	switch typ := typ.(type) {
	case *types.Basic:
		switch typ.Kind() {
		case types.UnsafePointer, types.String, types.UntypedNil, types.UntypedString:
			return true
		}
		return false

	case *types.Named:
		return typeHasPointers(typ.Underlying())

	case *types.Struct:
		for i := 0; i < typ.NumFields(); i++ {
			if typeHasPointers(typ.Field(i).Type()) {
				return true
			}
		}
		return false

	case *types.Array:
		return typeHasPointers(typ.Elem())

	default:
		return true
	}
}
