package ruleguard

import (
	"fmt"
	"go/build"
	"go/importer"
	"go/token"
	"go/types"
	"runtime"

	"github.com/quasilyte/go-ruleguard/internal/xsrcimporter"
)

// goImporter is a `types.Importer` that tries to load a package no matter what.
// It iterates through multiple import strategies and accepts whatever succeeds first.
type goImporter struct {
	// TODO(quasilyte): share importers with gogrep?

	state *engineState

	defaultImporter types.Importer
	srcImporter     types.Importer

	fset         *token.FileSet
	buildContext *build.Context

	debugImports bool
	debugPrint   func(string)
}

type goImporterConfig struct {
	fset         *token.FileSet
	debugImports bool
	debugPrint   func(string)
	buildContext *build.Context
}

func newGoImporter(state *engineState, config goImporterConfig) *goImporter {
	imp := &goImporter{
		state:           state,
		fset:            config.fset,
		debugImports:    config.debugImports,
		debugPrint:      config.debugPrint,
		defaultImporter: importer.Default(),
		buildContext:    config.buildContext,
	}
	imp.initSourceImporter()
	return imp
}

func (imp *goImporter) Import(path string) (*types.Package, error) {
	if pkg := imp.state.GetCachedPackage(path); pkg != nil {
		if imp.debugImports {
			imp.debugPrint(fmt.Sprintf(`imported "%s" from importer cache`, path))
		}
		return pkg, nil
	}

	pkg, srcErr := imp.srcImporter.Import(path)
	if srcErr == nil {
		imp.state.AddCachedPackage(path, pkg)
		if imp.debugImports {
			imp.debugPrint(fmt.Sprintf(`imported "%s" from source importer`, path))
		}
		return pkg, nil
	}

	pkg, defaultErr := imp.defaultImporter.Import(path)
	if defaultErr == nil {
		imp.state.AddCachedPackage(path, pkg)
		if imp.debugImports {
			imp.debugPrint(fmt.Sprintf(`imported "%s" from %s importer`, path, runtime.Compiler))
		}
		return pkg, nil
	}

	if imp.debugImports {
		imp.debugPrint(fmt.Sprintf(`failed to import "%s":`, path))
		imp.debugPrint(fmt.Sprintf("  %s importer: %v", runtime.Compiler, defaultErr))
		imp.debugPrint(fmt.Sprintf("  source importer: %v", srcErr))
		imp.debugPrint(fmt.Sprintf("  GOROOT=%q GOPATH=%q", imp.buildContext.GOROOT, imp.buildContext.GOPATH))
	}

	return nil, defaultErr
}

func (imp *goImporter) initSourceImporter() {
	if imp.buildContext == nil {
		if imp.debugImports {
			imp.debugPrint("using build.Default context")
		}
		imp.buildContext = &build.Default
	}
	imp.srcImporter = xsrcimporter.New(imp.buildContext, imp.fset)
}
