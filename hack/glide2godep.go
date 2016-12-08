//usr/bin/env go run $0 $@; exit
package main

import (
	"os"
	"fmt"
	"runtime"
	"encoding/json"

	"github.com/Masterminds/glide/cfg"
)

type Dependency struct {
	ImportPath string
	Rev        string
}

type Godeps struct {
	ImportPath   string
	GoVersion    string
	GodepVersion string
	Packages     []string `json:",omitempty"` // Arguments to save, if any.
	Deps         []Dependency
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, `glide2godep.go <glide.lock> <ImportPath>

Description: helps to emulate "godep restore" for glide based projects

Example: cd $GOPATH/src/github.com/coreos/etcd
         mkdir -p Godeps
         $GOPATH/src/k8s.io/kubernetes/hack/glide2godep.go glide.lock github.com/coreos/etcd > Godeps/Godeps.json
         godep restore -v
         cd -
         hack/godep-save.sh

Compare https://github.com/Masterminds/glide/issues/366 for "glide restore" support`)
		os.Exit(1)
	}

	glideLockPath, importPath := os.Args[1], os.Args[2]
	lf, err := cfg.ReadLockFile(glideLockPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to load glide lock file %q: %v\n", glideLockPath, err)
		os.Exit(1)
	}

	godeps := Godeps{
		ImportPath: importPath,
		GoVersion: runtime.Version(),
		GodepVersion: "v74", // version is private. This is good enough for now.
		Packages: []string{"./..."},
	}
	for _, imp := range append(lf.Imports, lf.DevImports...) {
		dep := Dependency{
			ImportPath: imp.Name,
			Rev: imp.Version,
		}
		godeps.Deps = append(godeps.Deps, dep)
	}
	out, err := json.MarshalIndent(godeps, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to convert to JSON: %v\n", err)
		os.Exit(1)
	}
	fmt.Println(string(out))
}
