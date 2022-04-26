package exhaustive

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/ast/inspector"
)

// constantValue is a constant.Value.ExactString().
type constantValue string

// Represents an enum type (or a potential enum type).
// It is a defined (named) type's name.
type enumType struct{ *types.TypeName }

func (et enumType) String() string           { return et.TypeName.String() } // for debugging
func (et enumType) scope() *types.Scope      { return et.TypeName.Parent() } // scope that the type is declared in
func (et enumType) factObject() types.Object { return et.TypeName }          // types.Object for fact export

// enumMembers is the members for a single enum type.
// The zero value is ready to use.
type enumMembers struct {
	Names        []string                   // enum member names, AST order
	NameToValue  map[string]constantValue   // enum member name -> constant value
	ValueToNames map[constantValue][]string // constant value -> enum member names
}

func (em *enumMembers) add(name string, val constantValue) {
	if em.NameToValue == nil {
		em.NameToValue = make(map[string]constantValue)
	}
	if em.ValueToNames == nil {
		em.ValueToNames = make(map[constantValue][]string)
	}

	em.Names = append(em.Names, name)
	em.NameToValue[name] = val
	em.ValueToNames[val] = append(em.ValueToNames[val], name)
}

func (em enumMembers) String() string { return em.factString() } // for debugging

func (em enumMembers) factString() string {
	var buf strings.Builder
	for j, vv := range em.Names {
		buf.WriteString(vv)
		// add comma separator between each enum member
		if j != len(em.Names)-1 {
			buf.WriteString(",")
		}
	}
	return buf.String()
}

func findEnums(pkgScopeOnly bool, pkg *types.Package, inspect *inspector.Inspector, info *types.Info) map[enumType]enumMembers {
	result := make(map[enumType]enumMembers)

	inspect.Preorder([]ast.Node{&ast.GenDecl{}}, func(n ast.Node) {
		gen := n.(*ast.GenDecl)
		if gen.Tok != token.CONST {
			return
		}
		for _, s := range gen.Specs {
			for _, name := range s.(*ast.ValueSpec).Names {
				enumTyp, memberName, val, ok := possibleEnumMember(name, info)
				if !ok {
					continue
				}
				if pkgScopeOnly && enumTyp.scope() != pkg.Scope() {
					continue
				}
				v := result[enumTyp]
				v.add(memberName, val)
				result[enumTyp] = v
			}
		}
	})

	return result
}

func possibleEnumMember(constName *ast.Ident, info *types.Info) (et enumType, name string, val constantValue, ok bool) {
	obj := info.Defs[constName]
	if obj == nil {
		panic(fmt.Sprintf("info.Defs[%s] == nil", constName))
	}
	if _, ok = obj.(*types.Const); !ok {
		panic(fmt.Sprintf("obj must be *types.Const, got %T", obj))
	}
	if isBlankIdentifier(obj) {
		// These objects have a nil parent scope.
		// Also, we have no real purpose to record them.
		return enumType{}, "", "", false
	}

	/*
		NOTE:

		type T int
		const A T = iota // obj.Type() is T

		type R T
		const B R = iota // obj.Type() is R

		type T2 int
		type T1 = T2
		const C T1 = iota // obj.Type() is T2

		type T3 = T4
		type T4 int
		type T5 = T3
		const D T5 = iota // obj.Type() is T4

		// And, in all these cases, validNamedBasic(obj.Type()) == true.
	*/

	if !validNamedBasic(obj.Type()) {
		return enumType{}, "", "", false
	}

	named := obj.Type().(*types.Named) // guaranteed by validNamedBasic()
	tn := named.Obj()

	// Enum type's scope and enum member's scope must be the same. If they're
	// not, don't consider the const a member. Additionally, the enum type and
	// the enum member must be in the same package (the scope check accounts for
	// this, too).
	if tn.Parent() != obj.Parent() {
		return enumType{}, "", "", false
	}

	return enumType{tn}, obj.Name(), determineConstVal(constName, info), true
}

func determineConstVal(name *ast.Ident, info *types.Info) constantValue {
	c := info.ObjectOf(name).(*types.Const)
	return constantValue(c.Val().ExactString())
}

func isBlankIdentifier(obj types.Object) bool {
	return obj.Name() == "_" // NOTE: go/types/decl.go does a direct comparison like this
}

func validBasic(basic *types.Basic) bool {
	switch i := basic.Info(); {
	case i&types.IsInteger != 0, i&types.IsFloat != 0, i&types.IsString != 0:
		return true
	}
	return false
}

// validNamedBasic returns whether the type t is a named type whose underlying
// type is a valid basic type to form an enum.
// A type that passes this check meets the definition of an enum type.
// Note that
//   validNamedBasic(t) == true => t.(*types.Named)
func validNamedBasic(t types.Type) bool {
	named, ok := t.(*types.Named)
	if !ok {
		return false
	}
	basic, ok := named.Underlying().(*types.Basic)
	if !ok || !validBasic(basic) {
		return false
	}
	return true
}
