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
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/packages"
)

var (
	verbose    = flag.Bool("verbose", false, "print more information")
	cross      = flag.Bool("cross", true, "build for all platforms")
	platforms  = flag.String("platform", "", "comma-separated list of platforms to typecheck")
	timings    = flag.Bool("time", false, "output times taken for each phase")
	defuses    = flag.Bool("defuse", false, "output defs/uses")
	serial     = flag.Bool("serial", false, "don't type check platforms in parallel (equivalent to --parallel=1)")
	parallel   = flag.Int("parallel", 3, "limits how many platforms can be checked in parallel. 0 means no limit.")
	skipTest   = flag.Bool("skip-test", false, "don't type check test code")
	tags       = flag.String("tags", "", "comma-separated list of build tags to apply in addition to go's defaults")
	ignoreDirs = flag.String("ignore-dirs", "", "comma-separated list of directories to ignore in addition to the default hardcoded list including staging, vendor, and hidden dirs")

	// When processed in order, windows and darwin are early to make
	// interesting OS-based errors happen earlier.
	crossPlatforms = []string{
		"linux/amd64", "windows/386",
		"darwin/amd64", "linux/arm",
		"linux/386", "windows/amd64",
		"linux/arm64", "linux/ppc64le",
		"linux/s390x",
	}

	// directories we always ignore
	standardIgnoreDirs = []string{
		// Staging code is symlinked from vendor/k8s.io, and uses import
		// paths as if it were inside of vendor/. It fails typechecking
		// inside of staging/, but works when typechecked as part of vendor/.
		"staging",
		// OS-specific vendor code tends to be imported by OS-specific
		// packages. We recursively typecheck imported vendored packages for
		// each OS, but don't typecheck everything for every OS.
		"vendor",
		"_output",
		// This is a weird one. /testdata/ is *mostly* ignored by Go,
		// and this translates to kubernetes/vendor not working.
		// edit/record.go doesn't compile without gopkg.in/yaml.v2
		// in $GOSRC/$GOROOT (both typecheck and the shell script).
		"pkg/kubectl/cmd/testdata/edit",
		// Tools we use for maintaining the code base but not necessarily
		// ship as part of the release
		"hack/tools",
	}
)

func newConfig(platform string) *packages.Config {
	platSplit := strings.Split(platform, "/")
	goos, goarch := platSplit[0], platSplit[1]
	mode := packages.NeedName | packages.NeedFiles | packages.NeedTypes | packages.NeedSyntax | packages.NeedDeps | packages.NeedImports
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

type collector struct {
	dirs       []string
	ignoreDirs []string
}

func newCollector(ignoreDirs string) collector {
	c := collector{
		ignoreDirs: append([]string(nil), standardIgnoreDirs...),
	}
	if ignoreDirs != "" {
		c.ignoreDirs = append(c.ignoreDirs, strings.Split(ignoreDirs, ",")...)
	}
	return c
}

func (c *collector) walk(roots []string) error {
	for _, root := range roots {
		err := filepath.Walk(root, c.handlePath)
		if err != nil {
			return err
		}
	}
	sort.Strings(c.dirs)
	return nil
}

// handlePath walks the filesystem recursively, collecting directories,
// ignoring some unneeded directories (hidden/vendored) that are handled
// specially later.
func (c *collector) handlePath(path string, info os.FileInfo, err error) error {
	if err != nil {
		return err
	}
	if info.IsDir() {
		name := info.Name()
		// Ignore hidden directories (.git, .cache, etc)
		if (len(name) > 1 && (name[0] == '.' || name[0] == '_')) || name == "testdata" {
			if *verbose {
				fmt.Printf("DBG: skipping dir %s\n", path)
			}
			return filepath.SkipDir
		}
		for _, dir := range c.ignoreDirs {
			if path == dir {
				if *verbose {
					fmt.Printf("DBG: ignoring dir %s\n", path)
				}
				return filepath.SkipDir
			}
		}
		// Make dirs into relative pkg names.
		// NOTE: can't use filepath.Join because it elides the leading "./"
		pkg := path
		if !strings.HasPrefix(pkg, "./") {
			pkg = "./" + pkg
		}
		c.dirs = append(c.dirs, pkg)
		if *verbose {
			fmt.Printf("DBG: added dir %s\n", path)
		}
	}
	return nil
}

func (c *collector) verify(plat string) ([]string, error) {
	errors := []packages.Error{}
	start := time.Now()
	config := newConfig(plat)

	rootPkgs, err := packages.Load(config, c.dirs...)
	if err != nil {
		return nil, err
	}

	// Recursively import all deps and flatten to one list.
	allMap := map[string]*packages.Package{}
	for _, pkg := range rootPkgs {
		if *verbose {
			serialFprintf(os.Stdout, "pkg %q has %d GoFiles\n", pkg.PkgPath, len(pkg.GoFiles))
		}
		allMap[pkg.PkgPath] = pkg
		if len(pkg.Imports) > 0 {
			for _, imp := range pkg.Imports {
				if *verbose {
					serialFprintf(os.Stdout, "pkg %q imports %q\n", pkg.PkgPath, imp.PkgPath)
				}
				allMap[imp.PkgPath] = imp
			}
		}
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
			if len(pkg.Errors) > 0 {
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

func serialFprintf(w io.Writer, format string, a ...interface{}) (n int, err error) {
	outMu.Lock()
	defer outMu.Unlock()
	return fmt.Fprintf(w, format, a...)
}

func main() {
	flag.Parse()
	args := flag.Args()

	if *verbose {
		*serial = true // to avoid confusing interleaved logs
	}

	if len(args) == 0 {
		args = append(args, ".")
	}

	c := newCollector(*ignoreDirs)

	if err := c.walk(args); err != nil {
		log.Fatalf("Error walking: %v", err)
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
			errors, err := c.verify(plat)
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
