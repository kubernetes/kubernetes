package ruleguard

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/types"

	"github.com/quasilyte/go-ruleguard/ruleguard/ir"
	"github.com/quasilyte/go-ruleguard/ruleguard/irconv"
)

func convertAST(ctx *LoadContext, imp *goImporter, filename string, src []byte) (*ir.File, *types.Package, error) {
	parserFlags := parser.ParseComments
	f, err := parser.ParseFile(ctx.Fset, filename, src, parserFlags)
	if err != nil {
		return nil, nil, fmt.Errorf("parse file error: %w", err)
	}

	typechecker := types.Config{Importer: imp}
	typesInfo := &types.Info{
		Types: map[ast.Expr]types.TypeAndValue{},
		Uses:  map[*ast.Ident]types.Object{},
		Defs:  map[*ast.Ident]types.Object{},
	}
	pkg, err := typechecker.Check("gorules", ctx.Fset, []*ast.File{f}, typesInfo)
	if err != nil {
		return nil, nil, fmt.Errorf("typechecker error: %w", err)
	}
	irconvCtx := &irconv.Context{
		Pkg:   pkg,
		Types: typesInfo,
		Fset:  ctx.Fset,
		Src:   src,
	}
	irfile, err := irconv.ConvertFile(irconvCtx, f)
	if err != nil {
		return nil, nil, fmt.Errorf("irconv error: %w", err)
	}
	return irfile, pkg, nil
}
