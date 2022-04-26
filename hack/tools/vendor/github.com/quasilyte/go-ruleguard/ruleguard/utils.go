package ruleguard

import (
	"go/ast"
	"go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"regexp/syntax"
	"strconv"
	"strings"
)

func regexpHasCaptureGroups(pattern string) bool {
	// regexp.Compile() uses syntax.Perl flags, so
	// we use the same flags here.
	re, err := syntax.Parse(pattern, syntax.Perl)
	if err != nil {
		return true // true is more conservative than false
	}

	found := false

	var walkRegexp func(*syntax.Regexp)
	walkRegexp = func(re *syntax.Regexp) {
		if found {
			return
		}
		// OpCapture handles both named and unnamed capture groups.
		if re.Op == syntax.OpCapture {
			found = true
			return
		}
		for _, sub := range re.Sub {
			walkRegexp(sub)
		}
	}
	walkRegexp(re)

	return found
}

func findDependency(pkg *types.Package, path string) *types.Package {
	if pkg.Path() == path {
		return pkg
	}
	// It looks like indirect dependencies are always incomplete?
	// If it's true, then we don't have to recurse here.
	for _, imported := range pkg.Imports() {
		if dep := findDependency(imported, path); dep != nil && dep.Complete() {
			return dep
		}
	}
	return nil
}

var typeByName = map[string]types.Type{
	// Predeclared types.
	`error`:      types.Universe.Lookup("error").Type(),
	`bool`:       types.Typ[types.Bool],
	`int`:        types.Typ[types.Int],
	`int8`:       types.Typ[types.Int8],
	`int16`:      types.Typ[types.Int16],
	`int32`:      types.Typ[types.Int32],
	`int64`:      types.Typ[types.Int64],
	`uint`:       types.Typ[types.Uint],
	`uint8`:      types.Typ[types.Uint8],
	`uint16`:     types.Typ[types.Uint16],
	`uint32`:     types.Typ[types.Uint32],
	`uint64`:     types.Typ[types.Uint64],
	`uintptr`:    types.Typ[types.Uintptr],
	`string`:     types.Typ[types.String],
	`float32`:    types.Typ[types.Float32],
	`float64`:    types.Typ[types.Float64],
	`complex64`:  types.Typ[types.Complex64],
	`complex128`: types.Typ[types.Complex128],

	// Predeclared aliases (provided for convenience).
	`byte`: types.Typ[types.Uint8],
	`rune`: types.Typ[types.Int32],
}

func typeFromString(s string) (types.Type, error) {
	s = strings.ReplaceAll(s, "?", "__any")

	n, err := parser.ParseExpr(s)
	if err != nil {
		return nil, err
	}
	return typeFromNode(n), nil
}

func typeFromNode(e ast.Expr) types.Type {
	switch e := e.(type) {
	case *ast.Ident:
		typ, ok := typeByName[e.Name]
		if ok {
			return typ
		}

	case *ast.ArrayType:
		elem := typeFromNode(e.Elt)
		if elem == nil {
			return nil
		}
		if e.Len == nil {
			return types.NewSlice(elem)
		}
		lit, ok := e.Len.(*ast.BasicLit)
		if !ok || lit.Kind != token.INT {
			return nil
		}
		length, err := strconv.Atoi(lit.Value)
		if err != nil {
			return nil
		}
		return types.NewArray(elem, int64(length))

	case *ast.MapType:
		keyType := typeFromNode(e.Key)
		if keyType == nil {
			return nil
		}
		valType := typeFromNode(e.Value)
		if valType == nil {
			return nil
		}
		return types.NewMap(keyType, valType)

	case *ast.StarExpr:
		typ := typeFromNode(e.X)
		if typ != nil {
			return types.NewPointer(typ)
		}

	case *ast.ParenExpr:
		return typeFromNode(e.X)

	case *ast.InterfaceType:
		if len(e.Methods.List) == 0 {
			return types.NewInterfaceType(nil, nil)
		}
	}

	return nil
}

func intValueOf(info *types.Info, expr ast.Expr) constant.Value {
	tv := info.Types[expr]
	if tv.Value == nil {
		return nil
	}
	if tv.Value.Kind() != constant.Int {
		return nil
	}
	return tv.Value
}

// isPure reports whether expr is a softly safe expression and contains
// no significant side-effects. As opposed to strictly safe expressions,
// soft safe expressions permit some forms of side-effects, like
// panic possibility during indexing or nil pointer dereference.
//
// Uses types info to determine type conversion expressions that
// are the only permitted kinds of call expressions.
// Note that is does not check whether called function really
// has any side effects. The analysis is very conservative.
func isPure(info *types.Info, expr ast.Expr) bool {
	// This list switch is not comprehensive and uses
	// whitelist to be on the conservative side.
	// Can be extended as needed.

	switch expr := expr.(type) {
	case *ast.StarExpr:
		return isPure(info, expr.X)
	case *ast.BinaryExpr:
		return isPure(info, expr.X) &&
			isPure(info, expr.Y)
	case *ast.UnaryExpr:
		return expr.Op != token.ARROW &&
			isPure(info, expr.X)
	case *ast.BasicLit, *ast.Ident, *ast.FuncLit:
		return true
	case *ast.IndexExpr:
		return isPure(info, expr.X) &&
			isPure(info, expr.Index)
	case *ast.SelectorExpr:
		return isPure(info, expr.X)
	case *ast.ParenExpr:
		return isPure(info, expr.X)
	case *ast.CompositeLit:
		return isPureList(info, expr.Elts)
	case *ast.CallExpr:
		return isTypeExpr(info, expr.Fun) && isPureList(info, expr.Args)

	default:
		return false
	}
}

// isPureList reports whether every expr in list is safe.
//
// See isPure.
func isPureList(info *types.Info, list []ast.Expr) bool {
	for _, expr := range list {
		if !isPure(info, expr) {
			return false
		}
	}
	return true
}

func isAddressable(info *types.Info, expr ast.Expr) bool {
	tv, ok := info.Types[expr]
	return ok && tv.Addressable()
}

func isConstant(info *types.Info, expr ast.Expr) bool {
	tv, ok := info.Types[expr]
	return ok && tv.Value != nil
}

func isConstantSlice(info *types.Info, expr ast.Expr) bool {
	switch expr := expr.(type) {
	case *ast.CallExpr:
		// Matches []byte("string").
		if len(expr.Args) != 1 {
			return false
		}
		lit, ok := expr.Args[0].(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return false
		}
		typ, ok := info.TypeOf(expr.Fun).(*types.Slice)
		if !ok {
			return false
		}
		basicType, ok := typ.Elem().(*types.Basic)
		return ok && basicType.Kind() == types.Uint8

	case *ast.CompositeLit:
		for _, elt := range expr.Elts {
			if !isConstant(info, elt) {
				return false
			}
		}
		return true

	default:
		return false
	}
}

// isTypeExpr reports whether x represents a type expression.
//
// Type expression does not evaluate to any run time value,
// but rather describes a type that is used inside Go expression.
//
// For example, (*T)(v) is a CallExpr that "calls" (*T).
// (*T) is a type expression that tells Go compiler type v should be converted to.
func isTypeExpr(info *types.Info, x ast.Expr) bool {
	switch x := x.(type) {
	case *ast.StarExpr:
		return isTypeExpr(info, x.X)
	case *ast.ParenExpr:
		return isTypeExpr(info, x.X)
	case *ast.SelectorExpr:
		return isTypeExpr(info, x.Sel)

	case *ast.Ident:
		// Identifier may be a type expression if object
		// it reffers to is a type name.
		_, ok := info.ObjectOf(x).(*types.TypeName)
		return ok

	case *ast.FuncType, *ast.StructType, *ast.InterfaceType, *ast.ArrayType, *ast.MapType, *ast.ChanType:
		return true

	default:
		return false
	}
}

func identOf(e ast.Expr) *ast.Ident {
	switch e := e.(type) {
	case *ast.ParenExpr:
		return identOf(e.X)
	case *ast.Ident:
		return e
	case *ast.SelectorExpr:
		return e.Sel
	default:
		return nil
	}
}
