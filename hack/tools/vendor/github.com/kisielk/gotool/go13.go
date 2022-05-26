// +build !go1.4

package gotool

import (
	"go/build"
	"path/filepath"
	"runtime"
)

var gorootSrc = filepath.Join(runtime.GOROOT(), "src", "pkg")

func shouldIgnoreImport(p *build.Package) bool {
	return true
}
