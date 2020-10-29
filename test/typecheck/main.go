/*
Copyright 2018 The Kubernetes Authors.

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

// do a fast type check of kubernetes code, for all platforms.
package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/packages"
)

var (
	verbose        = flag.Bool("verbose", false, "print more information")
	cross          = flag.Bool("cross", true, "build for all platforms")
	platforms      = flag.String("platform", "", "comma-separated list of platforms to typecheck")
	timings        = flag.Bool("time", false, "output times taken for each phase")
	defuses        = flag.Bool("defuse", false, "output defs/uses")
	serial         = flag.Bool("serial", false, "don't type check platforms in parallel (equivalent to --parallel=1)")
	parallel       = flag.Int("parallel", 2, "limits how many platforms can be checked in parallel. 0 means no limit.")
	skipTest       = flag.Bool("skip-test", false, "don't type check test code")
	tags           = flag.String("tags", "", "comma-separated list of build tags to apply in addition to go's defaults")
	ignorePatterns = flag.String("ignore", "", "comma-separated list of Go patterns to ignore")

	// When processed in order, windows and darwin are early to make
	// interesting OS-based errors happen earlier.
	crossPlatforms = []string{
		"linux/amd64", "windows/386",
		"darwin/amd64", "darwin/arm64",
		"linux/arm", "linux/386",
		"windows/amd64", "linux/arm64",
		"linux/ppc64le", "linux/s390x",
		"windows/arm64",
	}
)

func newConfig(platform string) *packages.Config {
	platSplit := strings.Split(platform, "/")
	goos, goarch := platSplit[0], platSplit[1]
	mode := packages.NeedName | packages.NeedFiles | packages.NeedTypes | packages.NeedSyntax | packages.NeedDeps | packages.NeedImports | packages.NeedModule
	if *defuses {
		mode = mode | packages.NeedTypesInfo
	}
	env := append(os.Environ(),
		// OpenShift doesn't build with CGO, since we use host-provided SSL
		// binaries for FIPS compatibility.
		// "CGO_ENABLED=1",
		fmt.Sprintf("GOOS=%s", goos),
		fmt.Sprintf("GOARCH=%s", goarch))
	tagstr := "selinux"
	if *tags != "" {
		tagstr = tagstr + "," + *tags
	}
	flags := []string{"-tags", tagstr}

	return &packages.Config{
		Mode:       mode,
		Env:        env,
		BuildFlags: flags,
		Tests:      !(*skipTest),
	}
}

func verify(plat string, patterns []string, ignore map[string]bool) ([]string, error) {
	errors := []packages.Error{}
	start := time.Now()
	config := newConfig(plat)

	pkgs, err := packages.Load(config, patterns...)
	if err != nil {
		return nil, err
	}

	// Recursively import all deps and flatten to one list.
	allMap := map[string]*packages.Package{}
	for _, pkg := range pkgs {
		if ignore[pkg.PkgPath] {
			continue
		}
		if *verbose {
			serialFprintf(os.Stdout, "pkg %q has %d GoFiles\n", pkg.PkgPath, len(pkg.GoFiles))
		}
		accumulate(pkg, allMap)
	}
	keys := make([]string, 0, len(allMap))
	for k := range allMap {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	allList := make([]*packages.Package, 0, len(keys))
	for _, k := range keys {
		allList = append(allList, allMap[k])
	}

	for _, pkg := range allList {
		if len(pkg.GoFiles) > 0 {
			if len(pkg.Errors) > 0 && (pkg.PkgPath == "main" || strings.Contains(pkg.PkgPath, ".")) {
				errors = append(errors, pkg.Errors...)
			}
		}
		if *defuses {
			for id, obj := range pkg.TypesInfo.Defs {
				serialFprintf(os.Stdout, "%s: %q defines %v\n",
					pkg.Fset.Position(id.Pos()), id.Name, obj)
			}
			for id, obj := range pkg.TypesInfo.Uses {
				serialFprintf(os.Stdout, "%s: %q uses %v\n",
					pkg.Fset.Position(id.Pos()), id.Name, obj)
			}
		}
	}
	if *timings {
		serialFprintf(os.Stdout, "%s took %.1fs\n", plat, time.Since(start).Seconds())
	}
	return dedup(errors), nil
}

func accumulate(pkg *packages.Package, allMap map[string]*packages.Package) {
	allMap[pkg.PkgPath] = pkg
	for _, imp := range pkg.Imports {
		if allMap[imp.PkgPath] != nil {
			continue
		}
		if *verbose {
			serialFprintf(os.Stdout, "pkg %q imports %q\n", pkg.PkgPath, imp.PkgPath)
		}
		accumulate(imp, allMap)
	}
}

func dedup(errors []packages.Error) []string {
	ret := []string{}

	m := map[string]bool{}
	for _, e := range errors {
		es := e.Error()
		if !m[es] {
			ret = append(ret, es)
			m[es] = true
		}
	}
	return ret
}

var outMu sync.Mutex

func serialFprintf(w io.Writer, format string, a ...interface{}) {
	outMu.Lock()
	defer outMu.Unlock()
	_, _ = fmt.Fprintf(w, format, a...)
}

func resolvePkgs(patterns ...string) (map[string]bool, error) {
	config := &packages.Config{
		Mode: packages.NeedName,
	}
	pkgs, err := packages.Load(config, patterns...)
	if err != nil {
		return nil, err
	}
	paths := map[string]bool{}
	for _, p := range pkgs {
		// ignore list errors (e.g. doesn't exist)
		if len(p.Errors) == 0 {
			paths[p.PkgPath] = true
		}
	}
	return paths, nil
}

func main() {
	flag.Parse()
	args := flag.Args()

	if *verbose {
		*serial = true // to avoid confusing interleaved logs
	}

	if len(args) == 0 {
		args = append(args, "./...")
	}

	ignore := []string{}
	if *ignorePatterns != "" {
		ignore = append(ignore, strings.Split(*ignorePatterns, ",")...)
	}
	ignorePkgs, err := resolvePkgs(ignore...)
	if err != nil {
		log.Fatalf("failed to resolve ignored packages: %v", err)
	}

	plats := crossPlatforms[:]
	if *platforms != "" {
		plats = strings.Split(*platforms, ",")
	} else if !*cross {
		plats = plats[:1]
	}

	var wg sync.WaitGroup
	var failMu sync.Mutex
	failed := false

	if *serial {
		*parallel = 1
	} else if *parallel == 0 {
		*parallel = len(plats)
	}
	throttle := make(chan int, *parallel)

	for _, plat := range plats {
		wg.Add(1)
		go func(plat string) {
			// block until there's room for this task
			throttle <- 1
			defer func() {
				// indicate this task is done
				<-throttle
			}()

			f := false
			serialFprintf(os.Stdout, "type-checking %s\n", plat)
			errors, err := verify(plat, args, ignorePkgs)
			if err != nil {
				serialFprintf(os.Stderr, "ERROR(%s): failed to verify: %v\n", plat, err)
				f = true
			} else if len(errors) > 0 {
				for _, e := range errors {
					// Special case CGo errors which may depend on headers we
					// don't have.
					if !strings.HasSuffix(e, "could not import C (no metadata for C)") {
						f = true
						serialFprintf(os.Stderr, "ERROR(%s): %s\n", plat, e)
					}
				}
			}
			failMu.Lock()
			failed = failed || f
			failMu.Unlock()
			wg.Done()
		}(plat)
	}
	wg.Wait()
	if failed {
		os.Exit(1)
	}
}
