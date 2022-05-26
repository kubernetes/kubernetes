package analysisutil

import (
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
)

var errType = types.Universe.Lookup("error").Type().Underlying().(*types.Interface)

// ImplementsError return whether t implements error interface.
func ImplementsError(t types.Type) bool {
	return types.Implements(t, errType)
}

// ObjectOf returns types.Object by given name in the package.
func ObjectOf(pass *analysis.Pass, pkg, name string) types.Object {
	obj := LookupFromImports(pass.Pkg.Imports(), pkg, name)
	if obj != nil {
		return obj
	}
	if RemoveVendor(pass.Pkg.Name()) != RemoveVendor(pkg) {
		return nil
	}
	return pass.Pkg.Scope().Lookup(name)
}

// TypeOf returns types.Type by given name in the package.
// TypeOf accepts pointer types such as *T.
func TypeOf(pass *analysis.Pass, pkg, name string) types.Type {
	if name == "" {
		return nil
	}

	if name[0] == '*' {
		obj := TypeOf(pass, pkg, name[1:])
		if obj == nil {
			return nil
		}
		return types.NewPointer(obj)
	}

	obj := ObjectOf(pass, pkg, name)
	if obj == nil {
		return nil
	}

	return obj.Type()
}

// MethodOf returns a method which has given name in the type.
func MethodOf(typ types.Type, name string) *types.Func {
	switch typ := typ.(type) {
	case *types.Named:
		for i := 0; i < typ.NumMethods(); i++ {
			if f := typ.Method(i); f.Name() == name {
				return f
			}
		}
	case *types.Pointer:
		return MethodOf(typ.Elem(), name)
	}
	return nil
}

// see: https://github.com/golang/go/issues/19670
func identical(x, y types.Type) (ret bool) {
	defer func() {
		r := recover()
		switch r := r.(type) {
		case string:
			if r == "unreachable" {
				ret = false
				return
			}
		case nil:
			return
		}
		panic(r)
	}()
	return types.Identical(x, y)
}

// Interfaces returns a map of interfaces which are declared in the package.
func Interfaces(pkg *types.Package) map[string]*types.Interface {
	ifs := map[string]*types.Interface{}

	for _, n := range pkg.Scope().Names() {
		o := pkg.Scope().Lookup(n)
		if o != nil {
			i, ok := o.Type().Underlying().(*types.Interface)
			if ok {
				ifs[n] = i
			}
		}
	}

	return ifs
}

// Structs returns a map of structs which are declared in the package.
func Structs(pkg *types.Package) map[string]*types.Struct {
	structs := map[string]*types.Struct{}

	for _, n := range pkg.Scope().Names() {
		o := pkg.Scope().Lookup(n)
		if o != nil {
			s, ok := o.Type().Underlying().(*types.Struct)
			if ok {
				structs[n] = s
			}
		}
	}

	return structs
}

// HasField returns whether the struct has the field.
func HasField(s *types.Struct, f *types.Var) bool {
	if s == nil || f == nil {
		return false
	}

	for i := 0; i < s.NumFields(); i++ {
		if s.Field(i) == f {
			return true
		}
	}

	return false
}

// Field returns field of the struct type.
// If the type is not struct or has not the field,
// Field returns -1, nil.
// If the type is a named type or a pointer type,
// Field calls itself recursively with
// an underlying type or an element type of pointer.
func Field(t types.Type, name string) (int, *types.Var) {
	switch t := t.(type) {
	case *types.Pointer:
		return Field(t.Elem(), name)
	case *types.Named:
		return Field(t.Underlying(), name)
	case *types.Struct:
		for i := 0; i < t.NumFields(); i++ {
			f := t.Field(i)
			if f.Name() == name {
				return i, f
			}
		}
	}

	return -1, nil
}

func TypesInfo(info ...*types.Info) *types.Info {
	if len(info) == 0 {
		return nil
	}

	var merged types.Info
	for i := range info {
		mergeTypesInfo(&merged, info[i])
	}

	return &merged
}

func mergeTypesInfo(i1, i2 *types.Info) {
	// Types
	if i1.Types == nil && i2.Types != nil {
		i1.Types = map[ast.Expr]types.TypeAndValue{}
	}
	for expr, tv := range i2.Types {
		i1.Types[expr] = tv
	}

	// Defs
	if i1.Defs == nil && i2.Defs != nil {
		i1.Defs = map[*ast.Ident]types.Object{}
	}
	for ident, obj := range i2.Defs {
		i1.Defs[ident] = obj
	}

	// Uses
	if i1.Uses == nil && i2.Uses != nil {
		i1.Uses = map[*ast.Ident]types.Object{}
	}
	for ident, obj := range i2.Uses {
		i1.Uses[ident] = obj
	}

	// Implicits
	if i1.Implicits == nil && i2.Implicits != nil {
		i1.Implicits = map[ast.Node]types.Object{}
	}
	for n, obj := range i2.Implicits {
		i1.Implicits[n] = obj
	}

	// Selections
	if i1.Selections == nil && i2.Selections != nil {
		i1.Selections = map[*ast.SelectorExpr]*types.Selection{}
	}
	for expr, sel := range i2.Selections {
		i1.Selections[expr] = sel
	}

	// Scopes
	if i1.Scopes == nil && i2.Scopes != nil {
		i1.Scopes = map[ast.Node]*types.Scope{}
	}
	for n, s := range i2.Scopes {
		i1.Scopes[n] = s
	}

	// InitOrder
	i1.InitOrder = append(i1.InitOrder, i2.InitOrder...)
}

// Under returns the most bottom underlying type.
// Deprecated: (types.Type).Underlying returns same value of it.
func Under(t types.Type) types.Type {
	return t.Underlying()
}
