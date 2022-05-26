package goutil

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/printer"
	"go/token"
	"go/types"
	"strings"
)

// SprintNode returns the textual representation of n.
// If fset is nil, freshly created file set will be used.
func SprintNode(fset *token.FileSet, n ast.Node) string {
	if fset == nil {
		fset = token.NewFileSet()
	}
	var buf strings.Builder
	if err := printer.Fprint(&buf, fset, n); err != nil {
		return ""
	}
	return buf.String()
}

type LoadConfig struct {
	Fset     *token.FileSet
	Filename string
	Data     interface{}
	Importer types.Importer
}

type LoadResult struct {
	Pkg    *types.Package
	Types  *types.Info
	Syntax *ast.File
}

func LoadGoFile(config LoadConfig) (*LoadResult, error) {
	imp := config.Importer
	if imp == nil {
		imp = importer.ForCompiler(config.Fset, "source", nil)
	}

	parserFlags := parser.ParseComments
	f, err := parser.ParseFile(config.Fset, config.Filename, config.Data, parserFlags)
	if err != nil {
		return nil, fmt.Errorf("parse file error: %w", err)
	}
	typechecker := types.Config{Importer: imp}
	typesInfo := &types.Info{
		Types: map[ast.Expr]types.TypeAndValue{},
		Uses:  map[*ast.Ident]types.Object{},
		Defs:  map[*ast.Ident]types.Object{},
	}
	pkg, err := typechecker.Check(f.Name.String(), config.Fset, []*ast.File{f}, typesInfo)
	if err != nil {
		return nil, fmt.Errorf("typechecker error: %w", err)
	}
	result := &LoadResult{
		Pkg:    pkg,
		Types:  typesInfo,
		Syntax: f,
	}
	return result, nil
}
