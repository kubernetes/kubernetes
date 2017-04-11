/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// teststale checks the staleness of a test binary. go test -c builds a test
// binary but it does no staleness check. In other words, every time one runs
// go test -c, it compiles the test packages and links the binary even when
// nothing has changed. This program helps to mitigate that problem by allowing
// to check the staleness of a given test package and its binary.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/golang/glog"
)

const usageHelp = "" +
	`This program checks the staleness of a given test package and its test
binary so that one can make a decision about re-building the test binary.

Usage:
  teststale -binary=/path/to/test/binary -package=package

Example:
  teststale -binary="$HOME/gosrc/bin/e2e.test" -package="k8s.io/kubernetes/test/e2e"

`

var (
	binary  = flag.String("binary", "", "filesystem path to the test binary file. Example: \"$HOME/gosrc/bin/e2e.test\"")
	pkgPath = flag.String("package", "", "import path of the test package in the format used while importing packages. Example: \"k8s.io/kubernetes/test/e2e\"")
)

func usage() {
	fmt.Fprintln(os.Stderr, usageHelp)
	fmt.Fprintln(os.Stderr, "Flags:")
	flag.PrintDefaults()
	os.Exit(2)
}

// golist is an interface emulating the `go list` command to get package information.
// TODO: Evaluate using `go/build` package instead. It doesn't provide staleness
// information, but we can probably run `go list` and `go/build.Import()` concurrently
// in goroutines and merge the results. Evaluate if that's faster.
type golist interface {
	pkgInfo(pkgPaths []string) ([]pkg, error)
}

// execmd implements the `golist` interface.
type execcmd struct {
	cmd  string
	args []string
	env  []string
}

func (e *execcmd) pkgInfo(pkgPaths []string) ([]pkg, error) {
	args := append(e.args, pkgPaths...)
	cmd := exec.Command(e.cmd, args...)
	cmd.Env = e.env

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to obtain the metadata output stream: %v", err)
	}

	dec := json.NewDecoder(stdout)

	// Start executing the command
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("command did not start: %v", err)
	}

	var pkgs []pkg
	for {
		var p pkg
		if err := dec.Decode(&p); err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata for package %s: %v", p.ImportPath, err)
		}
		pkgs = append(pkgs, p)
	}

	if err := cmd.Wait(); err != nil {
		return nil, fmt.Errorf("command did not complete: %v", err)
	}
	return pkgs, nil
}

type pkg struct {
	Dir          string
	ImportPath   string
	Target       string
	Stale        bool
	TestGoFiles  []string
	TestImports  []string
	XTestGoFiles []string
	XTestImports []string
}

func (p *pkg) isNewerThan(cmd golist, buildTime time.Time) bool {
	// If the package itself is stale, then we have to rebuild the whole thing anyway.
	if p.Stale {
		return true
	}

	// Test for file staleness
	for _, f := range p.TestGoFiles {
		if isNewerThan(filepath.Join(p.Dir, f), buildTime) {
			glog.V(4).Infof("test Go file %s is stale", f)
			return true
		}
	}
	for _, f := range p.XTestGoFiles {
		if isNewerThan(filepath.Join(p.Dir, f), buildTime) {
			glog.V(4).Infof("external test Go file %s is stale", f)
			return true
		}
	}

	imps := []string{}
	imps = append(imps, p.TestImports...)
	imps = append(imps, p.XTestImports...)

	// This calls `go list` the second time. This is required because the first
	// call to `go list` checks the staleness of the package in question by
	// looking the non-test dependencies, but it doesn't look at the test
	// dependencies. However, it returns the list of test dependencies. This
	// second call to `go list` checks the staleness of all the test
	// dependencies.
	pkgs, err := cmd.pkgInfo(imps)
	if err != nil || len(pkgs) < 1 {
		glog.V(4).Infof("failed to obtain metadata for packages %s: %v", imps, err)
		return true
	}

	for _, p := range pkgs {
		if p.Stale {
			glog.V(4).Infof("import %q is stale", p.ImportPath)
			return true
		}
	}

	return false
}

func isNewerThan(filename string, buildTime time.Time) bool {
	stat, err := os.Stat(filename)
	if err != nil {
		return true
	}
	return stat.ModTime().After(buildTime)
}

// isTestStale checks if the test binary is stale and needs to rebuilt.
// Some of the ideas here are inspired by how Go does staleness checks.
func isTestStale(cmd golist, binPath, pkgPath string) bool {
	bStat, err := os.Stat(binPath)
	if err != nil {
		glog.V(4).Infof("Couldn't obtain the modified time of the binary %s: %v", binPath, err)
		return true
	}
	buildTime := bStat.ModTime()

	pkgs, err := cmd.pkgInfo([]string{pkgPath})
	if err != nil || len(pkgs) < 1 {
		glog.V(4).Infof("Couldn't retrieve test package information for package %s: %v", pkgPath, err)
		return false
	}

	return pkgs[0].isNewerThan(cmd, buildTime)
}

func main() {
	flag.Usage = usage
	flag.Parse()

	cmd := &execcmd{
		cmd: "go",
		args: []string{
			"list",
			"-json",
		},
		env: os.Environ(),
	}
	if !isTestStale(cmd, *binary, *pkgPath) {
		os.Exit(1)
	}
}
