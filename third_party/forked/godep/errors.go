package main

import "errors"

var (
	errorLoadingDeps         = errors.New("error loading dependencies")
	errorLoadingPackages     = errors.New("error loading packages")
	errorCopyingSourceCode   = errors.New("error copying source code")
	errorNoPackagesUpdatable = errors.New("no packages can be updated")
)

type errPackageNotFound struct {
	path string
}

func (e errPackageNotFound) Error() string {
	return "Package (" + e.path + ") not found"
}
