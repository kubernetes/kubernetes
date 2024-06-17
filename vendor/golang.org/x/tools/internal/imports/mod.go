// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/gopathwalk"
	"golang.org/x/tools/internal/stdlib"
)

// Notes(rfindley): ModuleResolver appears to be heavily optimized for scanning
// as fast as possible, which is desirable for a call to goimports from the
// command line, but it doesn't work as well for gopls, where it suffers from
// slow startup (golang/go#44863) and intermittent hanging (golang/go#59216),
// both caused by populating the cache, albeit in slightly different ways.
//
// A high level list of TODOs:
//  - Optimize the scan itself, as there is some redundancy statting and
//    reading go.mod files.
//  - Invert the relationship between ProcessEnv and Resolver (see the
//    docstring of ProcessEnv).
//  - Make it easier to use an external resolver implementation.
//
// Smaller TODOs are annotated in the code below.

// ModuleResolver implements the Resolver interface for a workspace using
// modules.
//
// A goal of the ModuleResolver is to invoke the Go command as little as
// possible. To this end, it runs the Go command only for listing module
// information (i.e. `go list -m -e -json ...`). Package scanning, the process
// of loading package information for the modules, is implemented internally
// via the scan method.
//
// It has two types of state: the state derived from the go command, which
// is populated by init, and the state derived from scans, which is populated
// via scan. A root is considered scanned if it has been walked to discover
// directories. However, if the scan did not require additional information
// from the directory (such as package name or exports), the directory
// information itself may be partially populated. It will be lazily filled in
// as needed by scans, using the scanCallback.
type ModuleResolver struct {
	env *ProcessEnv

	// Module state, populated during construction
	dummyVendorMod *gocommand.ModuleJSON            // if vendoring is enabled, a pseudo-module to represent the /vendor directory
	moduleCacheDir string                           // GOMODCACHE, inferred from GOPATH if unset
	roots          []gopathwalk.Root                // roots to scan, in approximate order of importance
	mains          []*gocommand.ModuleJSON          // main modules
	mainByDir      map[string]*gocommand.ModuleJSON // module information by dir, to join with roots
	modsByModPath  []*gocommand.ModuleJSON          // all modules, ordered by # of path components in their module path
	modsByDir      []*gocommand.ModuleJSON          // ...or by the number of path components in their Dir.

	// Scanning state, populated by scan

	// scanSema prevents concurrent scans, and guards scannedRoots and the cache
	// fields below (though the caches themselves are concurrency safe).
	// Receive to acquire, send to release.
	scanSema     chan struct{}
	scannedRoots map[gopathwalk.Root]bool // if true, root has been walked

	// Caches of directory info, populated by scans and scan callbacks
	//
	// moduleCacheCache stores cached information about roots in the module
	// cache, which are immutable and therefore do not need to be invalidated.
	//
	// otherCache stores information about all other roots (even GOROOT), which
	// may change.
	moduleCacheCache *DirInfoCache
	otherCache       *DirInfoCache
}

// newModuleResolver returns a new module-aware goimports resolver.
//
// Note: use caution when modifying this constructor: changes must also be
// reflected in ModuleResolver.ClearForNewScan.
func newModuleResolver(e *ProcessEnv, moduleCacheCache *DirInfoCache) (*ModuleResolver, error) {
	r := &ModuleResolver{
		env:      e,
		scanSema: make(chan struct{}, 1),
	}
	r.scanSema <- struct{}{} // release

	goenv, err := r.env.goEnv()
	if err != nil {
		return nil, err
	}

	// TODO(rfindley): can we refactor to share logic with r.env.invokeGo?
	inv := gocommand.Invocation{
		BuildFlags: r.env.BuildFlags,
		ModFlag:    r.env.ModFlag,
		Env:        r.env.env(),
		Logf:       r.env.Logf,
		WorkingDir: r.env.WorkingDir,
	}

	vendorEnabled := false
	var mainModVendor *gocommand.ModuleJSON    // for module vendoring
	var mainModsVendor []*gocommand.ModuleJSON // for workspace vendoring

	goWork := r.env.Env["GOWORK"]
	if len(goWork) == 0 {
		// TODO(rfindley): VendorEnabled runs the go command to get GOFLAGS, but
		// they should be available from the ProcessEnv. Can we avoid the redundant
		// invocation?
		vendorEnabled, mainModVendor, err = gocommand.VendorEnabled(context.TODO(), inv, r.env.GocmdRunner)
		if err != nil {
			return nil, err
		}
	} else {
		vendorEnabled, mainModsVendor, err = gocommand.WorkspaceVendorEnabled(context.Background(), inv, r.env.GocmdRunner)
		if err != nil {
			return nil, err
		}
	}

	if vendorEnabled {
		if mainModVendor != nil {
			// Module vendor mode is on, so all the non-Main modules are irrelevant,
			// and we need to search /vendor for everything.
			r.mains = []*gocommand.ModuleJSON{mainModVendor}
			r.dummyVendorMod = &gocommand.ModuleJSON{
				Path: "",
				Dir:  filepath.Join(mainModVendor.Dir, "vendor"),
			}
			r.modsByModPath = []*gocommand.ModuleJSON{mainModVendor, r.dummyVendorMod}
			r.modsByDir = []*gocommand.ModuleJSON{mainModVendor, r.dummyVendorMod}
		} else {
			// Workspace vendor mode is on, so all the non-Main modules are irrelevant,
			// and we need to search /vendor for everything.
			r.mains = mainModsVendor
			r.dummyVendorMod = &gocommand.ModuleJSON{
				Path: "",
				Dir:  filepath.Join(filepath.Dir(goWork), "vendor"),
			}
			r.modsByModPath = append(append([]*gocommand.ModuleJSON{}, mainModsVendor...), r.dummyVendorMod)
			r.modsByDir = append(append([]*gocommand.ModuleJSON{}, mainModsVendor...), r.dummyVendorMod)
		}
	} else {
		// Vendor mode is off, so run go list -m ... to find everything.
		err := r.initAllMods()
		// We expect an error when running outside of a module with
		// GO111MODULE=on. Other errors are fatal.
		if err != nil {
			if errMsg := err.Error(); !strings.Contains(errMsg, "working directory is not part of a module") && !strings.Contains(errMsg, "go.mod file not found") {
				return nil, err
			}
		}
	}

	r.moduleCacheDir = gomodcacheForEnv(goenv)
	if r.moduleCacheDir == "" {
		return nil, fmt.Errorf("cannot resolve GOMODCACHE")
	}

	sort.Slice(r.modsByModPath, func(i, j int) bool {
		count := func(x int) int {
			return strings.Count(r.modsByModPath[x].Path, "/")
		}
		return count(j) < count(i) // descending order
	})
	sort.Slice(r.modsByDir, func(i, j int) bool {
		count := func(x int) int {
			return strings.Count(r.modsByDir[x].Dir, string(filepath.Separator))
		}
		return count(j) < count(i) // descending order
	})

	r.roots = []gopathwalk.Root{}
	if goenv["GOROOT"] != "" { // "" happens in tests
		r.roots = append(r.roots, gopathwalk.Root{Path: filepath.Join(goenv["GOROOT"], "/src"), Type: gopathwalk.RootGOROOT})
	}
	r.mainByDir = make(map[string]*gocommand.ModuleJSON)
	for _, main := range r.mains {
		r.roots = append(r.roots, gopathwalk.Root{Path: main.Dir, Type: gopathwalk.RootCurrentModule})
		r.mainByDir[main.Dir] = main
	}
	if vendorEnabled {
		r.roots = append(r.roots, gopathwalk.Root{Path: r.dummyVendorMod.Dir, Type: gopathwalk.RootOther})
	} else {
		addDep := func(mod *gocommand.ModuleJSON) {
			if mod.Replace == nil {
				// This is redundant with the cache, but we'll skip it cheaply enough
				// when we encounter it in the module cache scan.
				//
				// Including it at a lower index in r.roots than the module cache dir
				// helps prioritize matches from within existing dependencies.
				r.roots = append(r.roots, gopathwalk.Root{Path: mod.Dir, Type: gopathwalk.RootModuleCache})
			} else {
				r.roots = append(r.roots, gopathwalk.Root{Path: mod.Dir, Type: gopathwalk.RootOther})
			}
		}
		// Walk dependent modules before scanning the full mod cache, direct deps first.
		for _, mod := range r.modsByModPath {
			if !mod.Indirect && !mod.Main {
				addDep(mod)
			}
		}
		for _, mod := range r.modsByModPath {
			if mod.Indirect && !mod.Main {
				addDep(mod)
			}
		}
		// If provided, share the moduleCacheCache.
		//
		// TODO(rfindley): The module cache is immutable. However, the loaded
		// exports do depend on GOOS and GOARCH. Fortunately, the
		// ProcessEnv.buildContext does not adjust these from build.DefaultContext
		// (even though it should). So for now, this is OK to share, but we need to
		// add logic for handling GOOS/GOARCH.
		r.moduleCacheCache = moduleCacheCache
		r.roots = append(r.roots, gopathwalk.Root{Path: r.moduleCacheDir, Type: gopathwalk.RootModuleCache})
	}

	r.scannedRoots = map[gopathwalk.Root]bool{}
	if r.moduleCacheCache == nil {
		r.moduleCacheCache = NewDirInfoCache()
	}
	r.otherCache = NewDirInfoCache()
	return r, nil
}

// gomodcacheForEnv returns the GOMODCACHE value to use based on the given env
// map, which must have GOMODCACHE and GOPATH populated.
//
// TODO(rfindley): this is defensive refactoring.
//  1. Is this even relevant anymore? Can't we just read GOMODCACHE.
//  2. Use this to separate module cache scanning from other scanning.
func gomodcacheForEnv(goenv map[string]string) string {
	if gmc := goenv["GOMODCACHE"]; gmc != "" {
		return gmc
	}
	gopaths := filepath.SplitList(goenv["GOPATH"])
	if len(gopaths) == 0 {
		return ""
	}
	return filepath.Join(gopaths[0], "/pkg/mod")
}

func (r *ModuleResolver) initAllMods() error {
	stdout, err := r.env.invokeGo(context.TODO(), "list", "-m", "-e", "-json", "...")
	if err != nil {
		return err
	}
	for dec := json.NewDecoder(stdout); dec.More(); {
		mod := &gocommand.ModuleJSON{}
		if err := dec.Decode(mod); err != nil {
			return err
		}
		if mod.Dir == "" {
			if r.env.Logf != nil {
				r.env.Logf("module %v has not been downloaded and will be ignored", mod.Path)
			}
			// Can't do anything with a module that's not downloaded.
			continue
		}
		// golang/go#36193: the go command doesn't always clean paths.
		mod.Dir = filepath.Clean(mod.Dir)
		r.modsByModPath = append(r.modsByModPath, mod)
		r.modsByDir = append(r.modsByDir, mod)
		if mod.Main {
			r.mains = append(r.mains, mod)
		}
	}
	return nil
}

// ClearForNewScan invalidates the last scan.
//
// It preserves the set of roots, but forgets about the set of directories.
// Though it forgets the set of module cache directories, it remembers their
// contents, since they are assumed to be immutable.
func (r *ModuleResolver) ClearForNewScan() Resolver {
	<-r.scanSema // acquire r, to guard scannedRoots
	r2 := &ModuleResolver{
		env:            r.env,
		dummyVendorMod: r.dummyVendorMod,
		moduleCacheDir: r.moduleCacheDir,
		roots:          r.roots,
		mains:          r.mains,
		mainByDir:      r.mainByDir,
		modsByModPath:  r.modsByModPath,

		scanSema:         make(chan struct{}, 1),
		scannedRoots:     make(map[gopathwalk.Root]bool),
		otherCache:       NewDirInfoCache(),
		moduleCacheCache: r.moduleCacheCache,
	}
	r2.scanSema <- struct{}{} // r2 must start released
	// Invalidate root scans. We don't need to invalidate module cache roots,
	// because they are immutable.
	// (We don't support a use case where GOMODCACHE is cleaned in the middle of
	// e.g. a gopls session: the user must restart gopls to get accurate
	// imports.)
	//
	// Scanning for new directories in GOMODCACHE should be handled elsewhere,
	// via a call to ScanModuleCache.
	for _, root := range r.roots {
		if root.Type == gopathwalk.RootModuleCache && r.scannedRoots[root] {
			r2.scannedRoots[root] = true
		}
	}
	r.scanSema <- struct{}{} // release r
	return r2
}

// ClearModuleInfo invalidates resolver state that depends on go.mod file
// contents (essentially, the output of go list -m -json ...).
//
// Notably, it does not forget directory contents, which are reset
// asynchronously via ClearForNewScan.
//
// If the ProcessEnv is a GOPATH environment, ClearModuleInfo is a no op.
//
// TODO(rfindley): move this to a new env.go, consolidating ProcessEnv methods.
func (e *ProcessEnv) ClearModuleInfo() {
	if r, ok := e.resolver.(*ModuleResolver); ok {
		resolver, err := newModuleResolver(e, e.ModCache)
		if err != nil {
			e.resolver = nil
			e.resolverErr = err
			return
		}

		<-r.scanSema // acquire (guards caches)
		resolver.moduleCacheCache = r.moduleCacheCache
		resolver.otherCache = r.otherCache
		r.scanSema <- struct{}{} // release

		e.UpdateResolver(resolver)
	}
}

// UpdateResolver sets the resolver for the ProcessEnv to use in imports
// operations. Only for use with the result of [Resolver.ClearForNewScan].
//
// TODO(rfindley): this awkward API is a result of the (arguably) inverted
// relationship between configuration and state described in the doc comment
// for [ProcessEnv].
func (e *ProcessEnv) UpdateResolver(r Resolver) {
	e.resolver = r
	e.resolverErr = nil
}

// findPackage returns the module and directory from within the main modules
// and their dependencies that contains the package at the given import path,
// or returns nil, "" if no module is in scope.
func (r *ModuleResolver) findPackage(importPath string) (*gocommand.ModuleJSON, string) {
	// This can't find packages in the stdlib, but that's harmless for all
	// the existing code paths.
	for _, m := range r.modsByModPath {
		if !strings.HasPrefix(importPath, m.Path) {
			continue
		}
		pathInModule := importPath[len(m.Path):]
		pkgDir := filepath.Join(m.Dir, pathInModule)
		if r.dirIsNestedModule(pkgDir, m) {
			continue
		}

		if info, ok := r.cacheLoad(pkgDir); ok {
			if loaded, err := info.reachedStatus(nameLoaded); loaded {
				if err != nil {
					continue // No package in this dir.
				}
				return m, pkgDir
			}
			if scanned, err := info.reachedStatus(directoryScanned); scanned && err != nil {
				continue // Dir is unreadable, etc.
			}
			// This is slightly wrong: a directory doesn't have to have an
			// importable package to count as a package for package-to-module
			// resolution. package main or _test files should count but
			// don't.
			// TODO(heschi): fix this.
			if _, err := r.cachePackageName(info); err == nil {
				return m, pkgDir
			}
		}

		// Not cached. Read the filesystem.
		pkgFiles, err := os.ReadDir(pkgDir)
		if err != nil {
			continue
		}
		// A module only contains a package if it has buildable go
		// files in that directory. If not, it could be provided by an
		// outer module. See #29736.
		for _, fi := range pkgFiles {
			if ok, _ := r.env.matchFile(pkgDir, fi.Name()); ok {
				return m, pkgDir
			}
		}
	}
	return nil, ""
}

func (r *ModuleResolver) cacheLoad(dir string) (directoryPackageInfo, bool) {
	if info, ok := r.moduleCacheCache.Load(dir); ok {
		return info, ok
	}
	return r.otherCache.Load(dir)
}

func (r *ModuleResolver) cacheStore(info directoryPackageInfo) {
	if info.rootType == gopathwalk.RootModuleCache {
		r.moduleCacheCache.Store(info.dir, info)
	} else {
		r.otherCache.Store(info.dir, info)
	}
}

// cachePackageName caches the package name for a dir already in the cache.
func (r *ModuleResolver) cachePackageName(info directoryPackageInfo) (string, error) {
	if info.rootType == gopathwalk.RootModuleCache {
		return r.moduleCacheCache.CachePackageName(info)
	}
	return r.otherCache.CachePackageName(info)
}

func (r *ModuleResolver) cacheExports(ctx context.Context, env *ProcessEnv, info directoryPackageInfo) (string, []stdlib.Symbol, error) {
	if info.rootType == gopathwalk.RootModuleCache {
		return r.moduleCacheCache.CacheExports(ctx, env, info)
	}
	return r.otherCache.CacheExports(ctx, env, info)
}

// findModuleByDir returns the module that contains dir, or nil if no such
// module is in scope.
func (r *ModuleResolver) findModuleByDir(dir string) *gocommand.ModuleJSON {
	// This is quite tricky and may not be correct. dir could be:
	// - a package in the main module.
	// - a replace target underneath the main module's directory.
	//    - a nested module in the above.
	// - a replace target somewhere totally random.
	//    - a nested module in the above.
	// - in the mod cache.
	// - in /vendor/ in -mod=vendor mode.
	//    - nested module? Dunno.
	// Rumor has it that replace targets cannot contain other replace targets.
	//
	// Note that it is critical here that modsByDir is sorted to have deeper dirs
	// first. This ensures that findModuleByDir finds the innermost module.
	// See also golang/go#56291.
	for _, m := range r.modsByDir {
		if !strings.HasPrefix(dir, m.Dir) {
			continue
		}

		if r.dirIsNestedModule(dir, m) {
			continue
		}

		return m
	}
	return nil
}

// dirIsNestedModule reports if dir is contained in a nested module underneath
// mod, not actually in mod.
func (r *ModuleResolver) dirIsNestedModule(dir string, mod *gocommand.ModuleJSON) bool {
	if !strings.HasPrefix(dir, mod.Dir) {
		return false
	}
	if r.dirInModuleCache(dir) {
		// Nested modules in the module cache are pruned,
		// so it cannot be a nested module.
		return false
	}
	if mod != nil && mod == r.dummyVendorMod {
		// The /vendor pseudomodule is flattened and doesn't actually count.
		return false
	}
	modDir, _ := r.modInfo(dir)
	if modDir == "" {
		return false
	}
	return modDir != mod.Dir
}

func readModName(modFile string) string {
	modBytes, err := os.ReadFile(modFile)
	if err != nil {
		return ""
	}
	return modulePath(modBytes)
}

func (r *ModuleResolver) modInfo(dir string) (modDir, modName string) {
	if r.dirInModuleCache(dir) {
		if matches := modCacheRegexp.FindStringSubmatch(dir); len(matches) == 3 {
			index := strings.Index(dir, matches[1]+"@"+matches[2])
			modDir := filepath.Join(dir[:index], matches[1]+"@"+matches[2])
			return modDir, readModName(filepath.Join(modDir, "go.mod"))
		}
	}
	for {
		if info, ok := r.cacheLoad(dir); ok {
			return info.moduleDir, info.moduleName
		}
		f := filepath.Join(dir, "go.mod")
		info, err := os.Stat(f)
		if err == nil && !info.IsDir() {
			return dir, readModName(f)
		}

		d := filepath.Dir(dir)
		if len(d) >= len(dir) {
			return "", "" // reached top of file system, no go.mod
		}
		dir = d
	}
}

func (r *ModuleResolver) dirInModuleCache(dir string) bool {
	if r.moduleCacheDir == "" {
		return false
	}
	return strings.HasPrefix(dir, r.moduleCacheDir)
}

func (r *ModuleResolver) loadPackageNames(importPaths []string, srcDir string) (map[string]string, error) {
	names := map[string]string{}
	for _, path := range importPaths {
		// TODO(rfindley): shouldn't this use the dirInfoCache?
		_, packageDir := r.findPackage(path)
		if packageDir == "" {
			continue
		}
		name, err := packageDirToName(packageDir)
		if err != nil {
			continue
		}
		names[path] = name
	}
	return names, nil
}

func (r *ModuleResolver) scan(ctx context.Context, callback *scanCallback) error {
	ctx, done := event.Start(ctx, "imports.ModuleResolver.scan")
	defer done()

	processDir := func(info directoryPackageInfo) {
		// Skip this directory if we were not able to get the package information successfully.
		if scanned, err := info.reachedStatus(directoryScanned); !scanned || err != nil {
			return
		}
		pkg, err := r.canonicalize(info)
		if err != nil {
			return
		}
		if !callback.dirFound(pkg) {
			return
		}

		pkg.packageName, err = r.cachePackageName(info)
		if err != nil {
			return
		}
		if !callback.packageNameLoaded(pkg) {
			return
		}

		_, exports, err := r.loadExports(ctx, pkg, false)
		if err != nil {
			return
		}
		callback.exportsLoaded(pkg, exports)
	}

	// Start processing everything in the cache, and listen for the new stuff
	// we discover in the walk below.
	stop1 := r.moduleCacheCache.ScanAndListen(ctx, processDir)
	defer stop1()
	stop2 := r.otherCache.ScanAndListen(ctx, processDir)
	defer stop2()

	// We assume cached directories are fully cached, including all their
	// children, and have not changed. We can skip them.
	skip := func(root gopathwalk.Root, dir string) bool {
		if r.env.SkipPathInScan != nil && root.Type == gopathwalk.RootCurrentModule {
			if root.Path == dir {
				return false
			}

			if r.env.SkipPathInScan(filepath.Clean(dir)) {
				return true
			}
		}

		info, ok := r.cacheLoad(dir)
		if !ok {
			return false
		}
		// This directory can be skipped as long as we have already scanned it.
		// Packages with errors will continue to have errors, so there is no need
		// to rescan them.
		packageScanned, _ := info.reachedStatus(directoryScanned)
		return packageScanned
	}

	add := func(root gopathwalk.Root, dir string) {
		r.cacheStore(r.scanDirForPackage(root, dir))
	}

	// r.roots and the callback are not necessarily safe to use in the
	// goroutine below. Process them eagerly.
	roots := filterRoots(r.roots, callback.rootFound)
	// We can't cancel walks, because we need them to finish to have a usable
	// cache. Instead, run them in a separate goroutine and detach.
	scanDone := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			return
		case <-r.scanSema: // acquire
		}
		defer func() { r.scanSema <- struct{}{} }() // release
		// We have the lock on r.scannedRoots, and no other scans can run.
		for _, root := range roots {
			if ctx.Err() != nil {
				return
			}

			if r.scannedRoots[root] {
				continue
			}
			gopathwalk.WalkSkip([]gopathwalk.Root{root}, add, skip, gopathwalk.Options{Logf: r.env.Logf, ModulesEnabled: true})
			r.scannedRoots[root] = true
		}
		close(scanDone)
	}()
	select {
	case <-ctx.Done():
	case <-scanDone:
	}
	return nil
}

func (r *ModuleResolver) scoreImportPath(ctx context.Context, path string) float64 {
	if stdlib.HasPackage(path) {
		return MaxRelevance
	}
	mod, _ := r.findPackage(path)
	return modRelevance(mod)
}

func modRelevance(mod *gocommand.ModuleJSON) float64 {
	var relevance float64
	switch {
	case mod == nil: // out of scope
		return MaxRelevance - 4
	case mod.Indirect:
		relevance = MaxRelevance - 3
	case !mod.Main:
		relevance = MaxRelevance - 2
	default:
		relevance = MaxRelevance - 1 // main module ties with stdlib
	}

	_, versionString, ok := module.SplitPathVersion(mod.Path)
	if ok {
		index := strings.Index(versionString, "v")
		if index == -1 {
			return relevance
		}
		if versionNumber, err := strconv.ParseFloat(versionString[index+1:], 64); err == nil {
			relevance += versionNumber / 1000
		}
	}

	return relevance
}

// canonicalize gets the result of canonicalizing the packages using the results
// of initializing the resolver from 'go list -m'.
func (r *ModuleResolver) canonicalize(info directoryPackageInfo) (*pkg, error) {
	// Packages in GOROOT are already canonical, regardless of the std/cmd modules.
	if info.rootType == gopathwalk.RootGOROOT {
		return &pkg{
			importPathShort: info.nonCanonicalImportPath,
			dir:             info.dir,
			packageName:     path.Base(info.nonCanonicalImportPath),
			relevance:       MaxRelevance,
		}, nil
	}

	importPath := info.nonCanonicalImportPath
	mod := r.findModuleByDir(info.dir)
	// Check if the directory is underneath a module that's in scope.
	if mod != nil {
		// It is. If dir is the target of a replace directive,
		// our guessed import path is wrong. Use the real one.
		if mod.Dir == info.dir {
			importPath = mod.Path
		} else {
			dirInMod := info.dir[len(mod.Dir)+len("/"):]
			importPath = path.Join(mod.Path, filepath.ToSlash(dirInMod))
		}
	} else if !strings.HasPrefix(importPath, info.moduleName) {
		// The module's name doesn't match the package's import path. It
		// probably needs a replace directive we don't have.
		return nil, fmt.Errorf("package in %q is not valid without a replace statement", info.dir)
	}

	res := &pkg{
		importPathShort: importPath,
		dir:             info.dir,
		relevance:       modRelevance(mod),
	}
	// We may have discovered a package that has a different version
	// in scope already. Canonicalize to that one if possible.
	if _, canonicalDir := r.findPackage(importPath); canonicalDir != "" {
		res.dir = canonicalDir
	}
	return res, nil
}

func (r *ModuleResolver) loadExports(ctx context.Context, pkg *pkg, includeTest bool) (string, []stdlib.Symbol, error) {
	if info, ok := r.cacheLoad(pkg.dir); ok && !includeTest {
		return r.cacheExports(ctx, r.env, info)
	}
	return loadExportsFromFiles(ctx, r.env, pkg.dir, includeTest)
}

func (r *ModuleResolver) scanDirForPackage(root gopathwalk.Root, dir string) directoryPackageInfo {
	subdir := ""
	if dir != root.Path {
		subdir = dir[len(root.Path)+len("/"):]
	}
	importPath := filepath.ToSlash(subdir)
	if strings.HasPrefix(importPath, "vendor/") {
		// Only enter vendor directories if they're explicitly requested as a root.
		return directoryPackageInfo{
			status: directoryScanned,
			err:    fmt.Errorf("unwanted vendor directory"),
		}
	}
	switch root.Type {
	case gopathwalk.RootCurrentModule:
		importPath = path.Join(r.mainByDir[root.Path].Path, filepath.ToSlash(subdir))
	case gopathwalk.RootModuleCache:
		matches := modCacheRegexp.FindStringSubmatch(subdir)
		if len(matches) == 0 {
			return directoryPackageInfo{
				status: directoryScanned,
				err:    fmt.Errorf("invalid module cache path: %v", subdir),
			}
		}
		modPath, err := module.UnescapePath(filepath.ToSlash(matches[1]))
		if err != nil {
			if r.env.Logf != nil {
				r.env.Logf("decoding module cache path %q: %v", subdir, err)
			}
			return directoryPackageInfo{
				status: directoryScanned,
				err:    fmt.Errorf("decoding module cache path %q: %v", subdir, err),
			}
		}
		importPath = path.Join(modPath, filepath.ToSlash(matches[3]))
	}

	modDir, modName := r.modInfo(dir)
	result := directoryPackageInfo{
		status:                 directoryScanned,
		dir:                    dir,
		rootType:               root.Type,
		nonCanonicalImportPath: importPath,
		moduleDir:              modDir,
		moduleName:             modName,
	}
	if root.Type == gopathwalk.RootGOROOT {
		// stdlib packages are always in scope, despite the confusing go.mod
		return result
	}
	return result
}

// modCacheRegexp splits a path in a module cache into module, module version, and package.
var modCacheRegexp = regexp.MustCompile(`(.*)@([^/\\]*)(.*)`)

var (
	slashSlash = []byte("//")
	moduleStr  = []byte("module")
)

// modulePath returns the module path from the gomod file text.
// If it cannot find a module path, it returns an empty string.
// It is tolerant of unrelated problems in the go.mod file.
//
// Copied from cmd/go/internal/modfile.
func modulePath(mod []byte) string {
	for len(mod) > 0 {
		line := mod
		mod = nil
		if i := bytes.IndexByte(line, '\n'); i >= 0 {
			line, mod = line[:i], line[i+1:]
		}
		if i := bytes.Index(line, slashSlash); i >= 0 {
			line = line[:i]
		}
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, moduleStr) {
			continue
		}
		line = line[len(moduleStr):]
		n := len(line)
		line = bytes.TrimSpace(line)
		if len(line) == n || len(line) == 0 {
			continue
		}

		if line[0] == '"' || line[0] == '`' {
			p, err := strconv.Unquote(string(line))
			if err != nil {
				return "" // malformed quoted string or multiline module path
			}
			return p
		}

		return string(line)
	}
	return "" // missing module path
}
