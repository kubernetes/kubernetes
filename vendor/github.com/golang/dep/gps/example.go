// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"go/build"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/golang/dep/gps"
	"github.com/golang/dep/gps/pkgtree"
)

// This is probably the simplest possible implementation of gps. It does the
// substantive work that `go get` does, except:
//  1. It drops the resulting tree into vendor instead of GOPATH
//  2. It prefers semver tags (if available) over branches
//  3. It removes any vendor directories nested within dependencies
//
//  This will compile and work...and then blow away any vendor directory present
//  in the cwd. Be careful!
func main() {
	// Assume the current directory is correctly placed on a GOPATH, and that it's the
	// root of the project.
	root, _ := os.Getwd()
	srcprefix := filepath.Join(build.Default.GOPATH, "src") + string(filepath.Separator)
	importroot := filepath.ToSlash(strings.TrimPrefix(root, srcprefix))

	// Set up params, including tracing
	params := gps.SolveParameters{
		RootDir:         root,
		TraceLogger:     log.New(os.Stdout, "", 0),
		ProjectAnalyzer: NaiveAnalyzer{},
	}
	// Perform static analysis on the current project to find all of its imports.
	params.RootPackageTree, _ = pkgtree.ListPackages(root, importroot)

	// Set up a SourceManager. This manages interaction with sources (repositories).
	tempdir, _ := ioutil.TempDir("", "gps-repocache")
	sourcemgr, _ := gps.NewSourceManager(gps.SourceManagerConfig{Cachedir: filepath.Join(tempdir)})
	defer sourcemgr.Release()

	// Prep and run the solver
	solver, _ := gps.Prepare(params, sourcemgr)
	solution, err := solver.Solve()
	if err == nil {
		// If no failure, blow away the vendor dir and write a new one out,
		// stripping nested vendor directories as we go.
		os.RemoveAll(filepath.Join(root, "vendor"))
		gps.WriteDepTree(filepath.Join(root, "vendor"), solution, sourcemgr, true)
	}
}

// NaiveAnalyzer is a project analyzer that implements gps.ProjectAnalyzer interface.
type NaiveAnalyzer struct{}

// DeriveManifestAndLock is called when the solver needs manifest/lock data
// for a particular dependency project (identified by the gps.ProjectRoot
// parameter) at a particular version. That version will be checked out in a
// directory rooted at path.
func (a NaiveAnalyzer) DeriveManifestAndLock(path string, n gps.ProjectRoot) (gps.Manifest, gps.Lock, error) {
	return nil, nil, nil
}

// Info reports the name and version of the analyzer. This is used internally as part
// of gps' hashing memoization scheme.
func (a NaiveAnalyzer) Info() gps.ProjectAnalyzerInfo {
	return gps.ProjectAnalyzerInfo{
		Name:    "example-analyzer",
		Version: 1,
	}
}
