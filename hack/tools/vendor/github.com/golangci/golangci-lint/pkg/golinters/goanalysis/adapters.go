package goanalysis

import (
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/loader" //nolint:staticcheck // it's an adapter for golang.org/x/tools/go/packages
)

func MakeFakeLoaderProgram(pass *analysis.Pass) *loader.Program {
	prog := &loader.Program{
		Fset: pass.Fset,
		Created: []*loader.PackageInfo{
			{
				Pkg:                   pass.Pkg,
				Importable:            true, // not used
				TransitivelyErrorFree: true, // TODO

				Files:  pass.Files,
				Errors: nil,
				Info:   *pass.TypesInfo,
			},
		},
		AllPackages: map[*types.Package]*loader.PackageInfo{
			pass.Pkg: {
				Pkg:                   pass.Pkg,
				Importable:            true,
				TransitivelyErrorFree: true,
				Files:                 pass.Files,
				Errors:                nil,
				Info:                  *pass.TypesInfo,
			},
		},
	}
	return prog
}
