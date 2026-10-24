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
	"path/filepath"
	"slices"
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
	portable       = flag.String("portable", "", "comma-separated list of Go patterns whose dependencies must compile the same files on every platform (workspace modules only)")

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
		"CGO_ENABLED=1",
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

	if portability != nil {
		portability.record(plat, pkgs)
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

// portableAllowlist lists packages reachable from --portable that are known
// to compile different files per platform. Remove an entry once it is fixed;
// a stale entry fails the check.
var portableAllowlist = map[string]string{}

// portabilityCheck records which files each package compiles per platform.
// A binary built from a package with platform-specific files, such as
// foo_linux.go next to foo_unsupported.go, behaves differently depending on
// where it was built. e2e.test is built on macOS and Windows to test Linux
// clusters, so nothing it links may differ.
type portabilityCheck struct {
	roots  map[string]bool
	mu     sync.Mutex
	files  map[string]map[string][]string // platform -> package -> file names
	parent map[string]map[string]string   // platform -> package -> importer
}

var portability *portabilityCheck

func (c *portabilityCheck) record(plat string, pkgs []*packages.Package) {
	files := map[string][]string{}
	parent := map[string]string{}
	var queue []*packages.Package
	seen := map[string]bool{}
	visit := func(from string, pkg *packages.Package) {
		if seen[pkg.ID] {
			return
		}
		seen[pkg.ID] = true
		if _, ok := parent[pkg.PkgPath]; !ok {
			parent[pkg.PkgPath] = from
		}
		if pkg.Module != nil && pkg.Module.Main {
			names := make([]string, 0, len(pkg.GoFiles))
			for _, f := range pkg.GoFiles {
				names = append(names, filepath.Base(f))
			}
			sort.Strings(names)
			files[pkg.PkgPath] = names
		}
		queue = append(queue, pkg)
	}
	for _, pkg := range pkgs {
		if c.roots[strings.TrimSuffix(pkg.PkgPath, ".test")] {
			visit("", pkg)
		}
	}
	for ; len(queue) > 0; queue = queue[1:] {
		pkg := queue[0]
		paths := make([]string, 0, len(pkg.Imports))
		for path := range pkg.Imports {
			paths = append(paths, path)
		}
		sort.Strings(paths) // stable parents, stable chains
		for _, path := range paths {
			visit(pkg.PkgPath, pkg.Imports[path])
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.files[plat] = files
	c.parent[plat] = parent
}

// check reports packages whose files differ between platforms. It returns
// true on failure.
func (c *portabilityCheck) check(plats []string) bool {
	if len(plats) < 2 {
		return false
	}
	var differ []string
	for name := range c.files[plats[0]] {
		for _, plat := range plats[1:] {
			if !slices.Equal(c.files[plat][name], c.files[plats[0]][name]) {
				differ = append(differ, name)
				break
			}
		}
	}
	for _, plat := range plats[1:] {
		for name := range c.files[plat] {
			if _, ok := c.files[plats[0]][name]; !ok {
				differ = append(differ, name)
			}
		}
	}
	slices.Sort(differ)
	differ = slices.Compact(differ)

	failed := false
	for _, name := range differ {
		if _, ok := portableAllowlist[name]; ok {
			continue
		}
		failed = true
		// Files present on every platform are noise; show the rest.
		common := c.files[plats[0]][name]
		for _, plat := range plats[1:] {
			if f, ok := c.files[plat][name]; ok {
				common = slices.DeleteFunc(slices.Clone(common), func(s string) bool { return !slices.Contains(f, s) })
			}
		}
		serialFprintf(os.Stderr, "ERROR(portability): %s compiles different files per platform\n", name)
		for _, plat := range plats {
			f, ok := c.files[plat][name]
			if !ok {
				serialFprintf(os.Stderr, "    %s: not a dependency\n", plat)
				continue
			}
			only := slices.DeleteFunc(slices.Clone(f), func(s string) bool { return slices.Contains(common, s) })
			if len(only) == 0 {
				only = []string{"compiled"}
			}
			serialFprintf(os.Stderr, "    %s: %s\n", plat, strings.Join(only, ","))
		}
		chain := []string{name}
		for from := c.parent[plats[0]][name]; from != ""; from = c.parent[plats[0]][from] {
			chain = append([]string{from}, chain...)
		}
		serialFprintf(os.Stderr, "    via %s\n", strings.Join(chain, " -> "))
	}
	for name := range portableAllowlist {
		// A subset of platforms hides some differences; only the full list can prove an entry stale.
		if len(plats) == len(crossPlatforms) && !slices.Contains(differ, name) {
			failed = true
			serialFprintf(os.Stderr, "ERROR(portability): %s no longer differs, remove it from portableAllowlist\n", name)
		}
	}
	if failed {
		serialFprintf(os.Stderr, "!!! Packages linked into %s must compile the same files on every platform. Drop the dependency, or add it to portableAllowlist with a reason.\n", *portable)
	}
	return failed
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

	if *portable != "" {
		roots, err := resolvePkgs(strings.Split(*portable, ",")...)
		if err != nil {
			log.Fatalf("failed to resolve --portable packages: %v", err)
		}
		if len(roots) == 0 {
			log.Fatalf("--portable %q matched no packages", *portable)
		}
		portability = &portabilityCheck{roots: roots, files: map[string]map[string][]string{}, parent: map[string]map[string]string{}}
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
	if portability != nil && !failed {
		failed = portability.check(plats)
	}
	if failed {
		os.Exit(1)
	}
}
