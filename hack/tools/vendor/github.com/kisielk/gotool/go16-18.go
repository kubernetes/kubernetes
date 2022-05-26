// +build go1.6,!go1.9

package gotool

import (
	"go/build"
	"path/filepath"
	"runtime"
)

var gorootSrc = filepath.Join(runtime.GOROOT(), "src")

func shouldIgnoreImport(p *build.Package) bool {
	return p == nil || len(p.InvalidGoFiles) == 0
}
