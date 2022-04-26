package xsrcimporter

import (
	"go/build"
	"go/importer"
	"go/token"
	"go/types"
	"unsafe"
)

func New(ctxt *build.Context, fset *token.FileSet) types.Importer {
	imp := importer.ForCompiler(fset, "source", nil)
	ifaceVal := *(*iface)(unsafe.Pointer(&imp))
	srcImp := (*srcImporter)(ifaceVal.data)
	srcImp.ctxt = ctxt
	return imp
}

type iface struct {
	_    *byte
	data unsafe.Pointer
}

type srcImporter struct {
	ctxt *build.Context
	_    *token.FileSet
	_    types.Sizes
	_    map[string]*types.Package
}
