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

package varcheck

import (
	"flag"
	"go/ast"
	"go/token"
	"strings"

	"go/types"

	"golang.org/x/tools/go/loader"
)

var (
	buildTags = flag.String("varcheck.tags", "", "Build tags")
)

type object struct {
	pkgPath string
	name    string
}

type visitor struct {
	prog       *loader.Program
	pkg        *loader.PackageInfo
	uses       map[object]int
	positions  map[object]token.Position
	insideFunc bool
}

func getKey(obj types.Object) object {
	if obj == nil {
		return object{}
	}

	pkg := obj.Pkg()
	pkgPath := ""
	if pkg != nil {
		pkgPath = pkg.Path()
	}

	return object{
		pkgPath: pkgPath,
		name:    obj.Name(),
	}
}

func (v *visitor) decl(obj types.Object) {
	key := getKey(obj)
	if _, ok := v.uses[key]; !ok {
		v.uses[key] = 0
	}
	if _, ok := v.positions[key]; !ok {
		v.positions[key] = v.prog.Fset.Position(obj.Pos())
	}
}

func (v *visitor) use(obj types.Object) {
	key := getKey(obj)
	if _, ok := v.uses[key]; ok {
		v.uses[key]++
	} else {
		v.uses[key] = 1
	}
}

func isReserved(name string) bool {
	return name == "_" || strings.HasPrefix(strings.ToLower(name), "_cgo_")
}

func (v *visitor) Visit(node ast.Node) ast.Visitor {
	switch node := node.(type) {
	case *ast.Ident:
		v.use(v.pkg.Info.Uses[node])

	case *ast.ValueSpec:
		if !v.insideFunc {
			for _, ident := range node.Names {
				if !isReserved(ident.Name) {
					v.decl(v.pkg.Info.Defs[ident])
				}
			}
		}
		for _, val := range node.Values {
			ast.Walk(v, val)
		}
		if node.Type != nil {
			ast.Walk(v, node.Type)
		}
		return nil

	case *ast.FuncDecl:
		if node.Body != nil {
			v.insideFunc = true
			ast.Walk(v, node.Body)
			v.insideFunc = false
		}

		if node.Recv != nil {
			ast.Walk(v, node.Recv)
		}
		if node.Type != nil {
			ast.Walk(v, node.Type)
		}

		return nil
	}

	return v
}

type Issue struct {
	Pos     token.Position
	VarName string
}

func Run(program *loader.Program, reportExported bool) []Issue {
	var issues []Issue
	uses := make(map[object]int)
	positions := make(map[object]token.Position)

	for _, pkgInfo := range program.InitialPackages() {
		if pkgInfo.Pkg.Path() == "unsafe" {
			continue
		}

		v := &visitor{
			prog:      program,
			pkg:       pkgInfo,
			uses:      uses,
			positions: positions,
		}

		for _, f := range v.pkg.Files {
			ast.Walk(v, f)
		}
	}

	for obj, useCount := range uses {
		if useCount == 0 && (reportExported || !ast.IsExported(obj.name)) {
			pos := positions[obj]
			issues = append(issues, Issue{
				Pos:     pos,
				VarName: obj.name,
			})
		}
	}

	return issues
}
