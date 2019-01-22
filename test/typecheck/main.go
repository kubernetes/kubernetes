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
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/crypto/ssh/terminal"

	srcimporter "k8s.io/kubernetes/third_party/go-srcimporter"
)

var (
	verbose   = flag.Bool("verbose", false, "print more information")
	cross     = flag.Bool("cross", true, "build for all platforms")
	platforms = flag.String("platform", "", "comma-separated list of platforms to typecheck")
	timings   = flag.Bool("time", false, "output times taken for each phase")
	defuses   = flag.Bool("defuse", false, "output defs/uses")
	serial    = flag.Bool("serial", false, "don't type check platforms in parallel")

	isTerminal = terminal.IsTerminal(int(os.Stdout.Fd()))
	logPrefix  = ""

	// When processed in order, windows and darwin are early to make
	// interesting OS-based errors happen earlier.
	crossPlatforms = []string{
		"linux/amd64", "windows/386",
		"darwin/amd64", "linux/arm",
		"linux/386", "windows/amd64",
		"linux/arm64", "linux/ppc64le",
		"linux/s390x", "darwin/386",
	}
	darwinPlatString  = "darwin/386,darwin/amd64"
	windowsPlatString = "windows/386,windows/amd64"
)

type analyzer struct {
	fset      *token.FileSet // positions are relative to fset
	conf      types.Config
	ctx       build.Context
	failed    bool
	platform  string
	donePaths map[string]interface{}
	errors    []string
}

func newAnalyzer(platform string) *analyzer {
	ctx := build.Default
	platSplit := strings.Split(platform, "/")
	ctx.GOOS, ctx.GOARCH = platSplit[0], platSplit[1]
	ctx.CgoEnabled = true

	a := &analyzer{
		platform:  platform,
		fset:      token.NewFileSet(),
		ctx:       ctx,
		donePaths: make(map[string]interface{}),
	}
	a.conf = types.Config{
		FakeImportC: true,
		Error:       a.handleError,
		Sizes:       types.SizesFor("gc", a.ctx.GOARCH),
	}

	a.conf.Importer = srcimporter.New(
		&a.ctx, a.fset, make(map[string]*types.Package))

	if *verbose {
		fmt.Printf("context: %#v\n", ctx)
	}

	return a
}

func (a *analyzer) handleError(err error) {
	if e, ok := err.(types.Error); ok {
		// useful for some ignores:
		// path := e.Fset.Position(e.Pos).String()
		ignore := false
		// TODO(rmmh): read ignores from a file, so this code can
		// be Kubernetes-agnostic. Unused ignores should be treated as
		// errors, to ensure coverage isn't overly broad.
		if strings.Contains(e.Msg, "GetOpenAPIDefinitions") {
			// TODO(rmmh): figure out why this happens.
			// cmd/kube-apiserver/app/server.go:392:70
			// test/integration/framework/master_utils.go:131:84
			ignore = true
		}
		if ignore {
			if *verbose {
				fmt.Println("ignoring error:", err)
			}
			return
		}
	}
	a.errors = append(a.errors, err.Error())
	if *serial {
		fmt.Fprintf(os.Stderr, "%sERROR(%s) %s\n", logPrefix, a.platform, err)
	}
	a.failed = true
}

func (a *analyzer) dumpAndResetErrors() []string {
	es := a.errors
	a.errors = nil
	return es
}

// collect extracts test metadata from a file.
func (a *analyzer) collect(dir string) {
	if _, ok := a.donePaths[dir]; ok {
		return
	}
	a.donePaths[dir] = nil

	// Create the AST by parsing src.
	fs, err := parser.ParseDir(a.fset, dir, nil, parser.AllErrors)

	if err != nil {
		fmt.Println(logPrefix+"ERROR(syntax)", err)
		a.failed = true
		return
	}

	if len(fs) > 1 && *verbose {
		fmt.Println("multiple packages in dir:", dir)
	}

	for _, p := range fs {
		// returns first error, but a.handleError deals with it
		files := a.filterFiles(p.Files)
		if *verbose {
			fmt.Printf("path: %s package: %s files: ", dir, p.Name)
			for _, f := range files {
				fname := filepath.Base(a.fset.File(f.Pos()).Name())
				fmt.Printf("%s ", fname)
			}
			fmt.Printf("\n")
		}
		a.typeCheck(dir, files)
	}
}

// filterFiles restricts a list of files to only those that should be built by
// the current platform. This includes both build suffixes (_windows.go) and build
// tags ("// +build !linux" at the beginning).
func (a *analyzer) filterFiles(fs map[string]*ast.File) []*ast.File {
	files := []*ast.File{}
	for _, f := range fs {
		fpath := a.fset.File(f.Pos()).Name()
		dir, name := filepath.Split(fpath)
		matches, err := a.ctx.MatchFile(dir, name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%sERROR reading %s: %s\n", logPrefix, fpath, err)
			a.failed = true
			continue
		}
		if matches {
			files = append(files, f)
		}
	}
	return files
}

func (a *analyzer) typeCheck(dir string, files []*ast.File) error {
	info := types.Info{
		Defs: make(map[*ast.Ident]types.Object),
		Uses: make(map[*ast.Ident]types.Object),
	}

	// NOTE: this type check does a *recursive* import, but srcimporter
	// doesn't do a full type check (ignores function bodies)-- this has
	// some additional overhead.
	//
	// This means that we need to ensure that typeCheck runs on all
	// code we will be compiling.
	//
	// TODO(rmmh): Customize our forked srcimporter to do this better.
	pkg, err := a.conf.Check(dir, a.fset, files, &info)
	if err != nil {
		return err // type error
	}

	// A significant fraction of vendored code only compiles on Linux,
	// but it's only imported by code that has build-guards for Linux.
	// Track vendored code to type-check it in a second pass.
	for _, imp := range pkg.Imports() {
		if strings.HasPrefix(imp.Path(), "k8s.io/kubernetes/vendor/") {
			vendorPath := imp.Path()[len("k8s.io/kubernetes/"):]
			if *verbose {
				fmt.Println("recursively checking vendor path:", vendorPath)
			}
			a.collect(vendorPath)
		}
	}

	if *defuses {
		for id, obj := range info.Defs {
			fmt.Printf("%s: %q defines %v\n",
				a.fset.Position(id.Pos()), id.Name, obj)
		}
		for id, obj := range info.Uses {
			fmt.Printf("%s: %q uses %v\n",
				a.fset.Position(id.Pos()), id.Name, obj)
		}
	}

	return nil
}

type collector struct {
	dirs []string
}

// handlePath walks the filesystem recursively, collecting directories,
// ignoring some unneeded directories (hidden/vendored) that are handled
// specially later.
func (c *collector) handlePath(path string, info os.FileInfo, err error) error {
	if err != nil {
		return err
	}
	if info.IsDir() {
		// Ignore hidden directories (.git, .cache, etc)
		if len(path) > 1 && path[0] == '.' ||
			// Staging code is symlinked from vendor/k8s.io, and uses import
			// paths as if it were inside of vendor/. It fails typechecking
			// inside of staging/, but works when typechecked as part of vendor/.
			path == "staging" ||
			// OS-specific vendor code tends to be imported by OS-specific
			// packages. We recursively typecheck imported vendored packages for
			// each OS, but don't typecheck everything for every OS.
			path == "vendor" ||
			path == "_output" ||
			// This is a weird one. /testdata/ is *mostly* ignored by Go,
			// and this translates to kubernetes/vendor not working.
			// edit/record.go doesn't compile without gopkg.in/yaml.v2
			// in $GOSRC/$GOROOT (both typecheck and the shell script).
			path == "pkg/kubectl/cmd/testdata/edit" {
			return filepath.SkipDir
		}
		c.dirs = append(c.dirs, path)
	}
	return nil
}

type analyzerResult struct {
	platform string
	dir      string
	errors   []string
}

func dedupeErrors(out io.Writer, results chan analyzerResult, nDirs, nPlatforms int) {
	pkgRes := make(map[string][]analyzerResult)
	for done := 0; done < nDirs; {
		res := <-results
		pkgRes[res.dir] = append(pkgRes[res.dir], res)
		if len(pkgRes[res.dir]) != nPlatforms {
			continue // expect more results for dir
		}
		done++
		// Collect list of platforms for each error
		errPlats := map[string][]string{}
		for _, res := range pkgRes[res.dir] {
			for _, err := range res.errors {
				errPlats[err] = append(errPlats[err], res.platform)
			}
		}
		// Print each error (in the same order!) once.
		for _, res := range pkgRes[res.dir] {
			for _, err := range res.errors {
				if errPlats[err] == nil {
					continue // already printed
				}
				sort.Strings(errPlats[err])
				plats := strings.Join(errPlats[err], ",")
				if len(errPlats[err]) == len(crossPlatforms) {
					plats = "all"
				} else if plats == darwinPlatString {
					plats = "darwin"
				} else if plats == windowsPlatString {
					plats = "windows"
				}
				fmt.Fprintf(out, "%sERROR(%s) %s\n", logPrefix, plats, err)
				delete(errPlats, err)
			}
		}
		delete(pkgRes, res.dir)
	}
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

	c := collector{}
	for _, arg := range args {
		err := filepath.Walk(arg, c.handlePath)
		if err != nil {
			log.Fatalf("Error walking: %v", err)
		}
	}
	sort.Strings(c.dirs)

	ps := crossPlatforms[:]
	if *platforms != "" {
		ps = strings.Split(*platforms, ",")
	} else if !*cross {
		ps = ps[:1]
	}

	fmt.Println("type-checking: ", strings.Join(ps, ", "))

	var wg sync.WaitGroup
	var processedDirs int64
	var currentWork int64 // (dir_index << 8) | platform_index
	statuses := make([]int, len(ps))
	var results chan analyzerResult
	if !*serial {
		results = make(chan analyzerResult)
		wg.Add(1)
		go func() {
			dedupeErrors(os.Stderr, results, len(c.dirs), len(ps))
			wg.Done()
		}()
	}
	for i, p := range ps {
		wg.Add(1)
		fn := func(i int, p string) {
			start := time.Now()
			a := newAnalyzer(p)
			for n, dir := range c.dirs {
				a.collect(dir)
				atomic.AddInt64(&processedDirs, 1)
				atomic.StoreInt64(&currentWork, int64(n<<8|i))
				if results != nil {
					results <- analyzerResult{p, dir, a.dumpAndResetErrors()}
				}
			}
			if a.failed {
				statuses[i] = 1
			}
			if *timings {
				fmt.Printf("%s took %.1fs\n", p, time.Since(start).Seconds())
			}
			wg.Done()
		}
		if *serial {
			fn(i, p)
		} else {
			go fn(i, p)
		}
	}
	if isTerminal {
		logPrefix = "\r" // clear status bar when printing
		// Display a status bar so devs can estimate completion times.
		wg.Add(1)
		go func() {
			total := len(ps) * len(c.dirs)
			for proc := 0; ; proc = int(atomic.LoadInt64(&processedDirs)) {
				work := atomic.LoadInt64(&currentWork)
				dir := c.dirs[work>>8]
				platform := ps[work&0xFF]
				if len(dir) > 80 {
					dir = dir[:80]
				}
				fmt.Printf("\r%d/%d \033[2m%-13s\033[0m %-80s", proc, total, platform, dir)
				if proc == total {
					fmt.Println()
					break
				}
				time.Sleep(50 * time.Millisecond)
			}
			wg.Done()
		}()
	}
	wg.Wait()
	for _, status := range statuses {
		if status != 0 {
			os.Exit(status)
		}
	}
}
