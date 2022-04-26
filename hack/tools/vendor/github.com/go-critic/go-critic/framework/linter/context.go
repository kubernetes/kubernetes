package linter

import (
	"go/ast"
	"go/types"
	"strconv"
)

func resolvePkgObjects(ctx *Context, f *ast.File) {
	ctx.PkgObjects = make(map[*types.PkgName]string, len(f.Imports))

	for _, spec := range f.Imports {
		if spec.Name != nil {
			obj := ctx.TypesInfo.ObjectOf(spec.Name)
			ctx.PkgObjects[obj.(*types.PkgName)] = spec.Name.Name
		} else {
			obj := ctx.TypesInfo.Implicits[spec]
			ctx.PkgObjects[obj.(*types.PkgName)] = obj.Name()
		}
	}
}

func resolvePkgRenames(ctx *Context, f *ast.File) {
	ctx.PkgRenames = make(map[string]string)

	for _, spec := range f.Imports {
		if spec.Name != nil {
			path, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				panic(err)
			}
			ctx.PkgRenames[path] = spec.Name.Name
		}
	}
}
