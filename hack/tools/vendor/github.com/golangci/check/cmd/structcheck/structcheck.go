// structcheck
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

package structcheck

import (
	"flag"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/loader"
)

var (
	assignmentsOnly = flag.Bool("structcheck.a", false, "Count assignments only")
	loadTestFiles   = flag.Bool("structcheck.t", false, "Load test files too")
	buildTags       = flag.String("structcheck.tags", "", "Build tags")
)

type visitor struct {
	prog *loader.Program
	pkg  *loader.PackageInfo
	m    map[types.Type]map[string]int
	skip map[types.Type]struct{}
}

func (v *visitor) decl(t types.Type, fieldName string) {
	if _, ok := v.m[t]; !ok {
		v.m[t] = make(map[string]int)
	}
	if _, ok := v.m[t][fieldName]; !ok {
		v.m[t][fieldName] = 0
	}
}

func (v *visitor) assignment(t types.Type, fieldName string) {
	if _, ok := v.m[t]; !ok {
		v.m[t] = make(map[string]int)
	}
	if _, ok := v.m[t][fieldName]; ok {
		v.m[t][fieldName]++
	} else {
		v.m[t][fieldName] = 1
	}
}

func (v *visitor) typeSpec(node *ast.TypeSpec) {
	if strukt, ok := node.Type.(*ast.StructType); ok {
		t := v.pkg.Info.Defs[node.Name].Type()
		for _, f := range strukt.Fields.List {
			if len(f.Names) > 0 {
				fieldName := f.Names[0].Name
				v.decl(t, fieldName)
			}
		}
	}
}

func (v *visitor) typeAndFieldName(expr *ast.SelectorExpr) (types.Type, string, bool) {
	selection := v.pkg.Info.Selections[expr]
	if selection == nil {
		return nil, "", false
	}
	recv := selection.Recv()
	if ptr, ok := recv.(*types.Pointer); ok {
		recv = ptr.Elem()
	}
	return recv, selection.Obj().Name(), true
}

func (v *visitor) assignStmt(node *ast.AssignStmt) {
	for _, lhs := range node.Lhs {
		var selector *ast.SelectorExpr
		switch expr := lhs.(type) {
		case *ast.SelectorExpr:
			selector = expr
		case *ast.IndexExpr:
			if expr, ok := expr.X.(*ast.SelectorExpr); ok {
				selector = expr
			}
		}
		if selector != nil {
			if t, fn, ok := v.typeAndFieldName(selector); ok {
				v.assignment(t, fn)
			}
		}
	}
}

func (v *visitor) compositeLiteral(node *ast.CompositeLit) {
	t := v.pkg.Info.Types[node.Type].Type
	for _, expr := range node.Elts {
		if kv, ok := expr.(*ast.KeyValueExpr); ok {
			if ident, ok := kv.Key.(*ast.Ident); ok {
				v.assignment(t, ident.Name)
			}
		} else {
			// Struct literal with positional values.
			// All the fields are assigned.
			v.skip[t] = struct{}{}
			break
		}
	}
}

func (v *visitor) Visit(node ast.Node) ast.Visitor {
	switch node := node.(type) {
	case *ast.TypeSpec:
		v.typeSpec(node)

	case *ast.AssignStmt:
		if *assignmentsOnly {
			v.assignStmt(node)
		}

	case *ast.SelectorExpr:
		if !*assignmentsOnly {
			if t, fn, ok := v.typeAndFieldName(node); ok {
				v.assignment(t, fn)
			}
		}

	case *ast.CompositeLit:
		v.compositeLiteral(node)
	}

	return v
}

type Issue struct {
	Pos       token.Position
	Type      string
	FieldName string
}

func Run(program *loader.Program, reportExported bool) []Issue {
	var issues []Issue
	for _, pkg := range program.InitialPackages() {
		visitor := &visitor{
			m:    make(map[types.Type]map[string]int),
			skip: make(map[types.Type]struct{}),
			prog: program,
			pkg:  pkg,
		}
		for _, f := range pkg.Files {
			ast.Walk(visitor, f)
		}

		for t := range visitor.m {
			if _, skip := visitor.skip[t]; skip {
				continue
			}
			for fieldName, v := range visitor.m[t] {
				if !reportExported && ast.IsExported(fieldName) {
					continue
				}
				if v == 0 {
					field, _, _ := types.LookupFieldOrMethod(t, false, pkg.Pkg, fieldName)
					if field == nil {
						fmt.Printf("%s: unknown field or method: %s.%s\n", pkg.Pkg.Path(), t, fieldName)
						continue
					}
					if fieldName == "XMLName" {
						if named, ok := field.Type().(*types.Named); ok && named.Obj().Pkg().Path() == "encoding/xml" {
							continue
						}
					}
					pos := program.Fset.Position(field.Pos())
					issues = append(issues, Issue{
						Pos:       pos,
						Type:      types.TypeString(t, nil),
						FieldName: fieldName,
					})
				}
			}
		}
	}

	return issues
}
