// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"context"
	"fmt"
	"path"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/mod/module"
	"golang.org/x/tools/internal/gopathwalk"
)

// To find packages to import, the resolver needs to know about all of
// the packages that could be imported. This includes packages that are
// already in modules that are in (1) the current module, (2) replace targets,
// and (3) packages in the module cache. Packages in (1) and (2) may change over
// time, as the client may edit the current module and locally replaced modules.
// The module cache (which includes all of the packages in (3)) can only
// ever be added to.
//
// The resolver can thus save state about packages in the module cache
// and guarantee that this will not change over time. To obtain information
// about new modules added to the module cache, the module cache should be
// rescanned.
//
// It is OK to serve information about modules that have been deleted,
// as they do still exist.
// TODO(suzmue): can we share information with the caller about
// what module needs to be downloaded to import this package?

type directoryPackageStatus int

const (
	_ directoryPackageStatus = iota
	directoryScanned
	nameLoaded
	exportsLoaded
)

// directoryPackageInfo holds (possibly incomplete) information about packages
// contained in a given directory.
type directoryPackageInfo struct {
	// status indicates the extent to which this struct has been filled in.
	status directoryPackageStatus
	// err is non-nil when there was an error trying to reach status.
	err error

	// Set when status >= directoryScanned.

	// dir is the absolute directory of this package.
	dir      string
	rootType gopathwalk.RootType
	// nonCanonicalImportPath is the package's expected import path. It may
	// not actually be importable at that path.
	nonCanonicalImportPath string

	// Module-related information.
	moduleDir  string // The directory that is the module root of this dir.
	moduleName string // The module name that contains this dir.

	// Set when status >= nameLoaded.

	packageName string // the package name, as declared in the source.

	// Set when status >= exportsLoaded.
	// TODO(rfindley): it's hard to see this, but exports depend implicitly on
	// the default build context GOOS and GOARCH.
	//
	// We can make this explicit, and key exports by GOOS, GOARCH.
	exports []string
}

// reachedStatus returns true when info has a status at least target and any error associated with
// an attempt to reach target.
func (info *directoryPackageInfo) reachedStatus(target directoryPackageStatus) (bool, error) {
	if info.err == nil {
		return info.status >= target, nil
	}
	if info.status == target {
		return true, info.err
	}
	return true, nil
}

// DirInfoCache is a concurrency-safe map for storing information about
// directories that may contain packages.
//
// The information in this cache is built incrementally. Entries are initialized in scan.
// No new keys should be added in any other functions, as all directories containing
// packages are identified in scan.
//
// Other functions, including loadExports and findPackage, may update entries in this cache
// as they discover new things about the directory.
//
// The information in the cache is not expected to change for the cache's
// lifetime, so there is no protection against competing writes. Users should
// take care not to hold the cache across changes to the underlying files.
type DirInfoCache struct {
	mu sync.Mutex
	// dirs stores information about packages in directories, keyed by absolute path.
	dirs      map[string]*directoryPackageInfo
	listeners map[*int]cacheListener
}

func NewDirInfoCache() *DirInfoCache {
	return &DirInfoCache{
		dirs:      make(map[string]*directoryPackageInfo),
		listeners: make(map[*int]cacheListener),
	}
}

type cacheListener func(directoryPackageInfo)

// ScanAndListen calls listener on all the items in the cache, and on anything
// newly added. The returned stop function waits for all in-flight callbacks to
// finish and blocks new ones.
func (d *DirInfoCache) ScanAndListen(ctx context.Context, listener cacheListener) func() {
	ctx, cancel := context.WithCancel(ctx)

	// Flushing out all the callbacks is tricky without knowing how many there
	// are going to be. Setting an arbitrary limit makes it much easier.
	const maxInFlight = 10
	sema := make(chan struct{}, maxInFlight)
	for i := 0; i < maxInFlight; i++ {
		sema <- struct{}{}
	}

	cookie := new(int) // A unique ID we can use for the listener.

	// We can't hold mu while calling the listener.
	d.mu.Lock()
	var keys []string
	for key := range d.dirs {
		keys = append(keys, key)
	}
	d.listeners[cookie] = func(info directoryPackageInfo) {
		select {
		case <-ctx.Done():
			return
		case <-sema:
		}
		listener(info)
		sema <- struct{}{}
	}
	d.mu.Unlock()

	stop := func() {
		cancel()
		d.mu.Lock()
		delete(d.listeners, cookie)
		d.mu.Unlock()
		for i := 0; i < maxInFlight; i++ {
			<-sema
		}
	}

	// Process the pre-existing keys.
	for _, k := range keys {
		select {
		case <-ctx.Done():
			return stop
		default:
		}
		if v, ok := d.Load(k); ok {
			listener(v)
		}
	}

	return stop
}

// Store stores the package info for dir.
func (d *DirInfoCache) Store(dir string, info directoryPackageInfo) {
	d.mu.Lock()
	// TODO(rfindley, golang/go#59216): should we overwrite an existing entry?
	// That seems incorrect as the cache should be idempotent.
	_, old := d.dirs[dir]
	d.dirs[dir] = &info
	var listeners []cacheListener
	for _, l := range d.listeners {
		listeners = append(listeners, l)
	}
	d.mu.Unlock()

	if !old {
		for _, l := range listeners {
			l(info)
		}
	}
}

// Load returns a copy of the directoryPackageInfo for absolute directory dir.
func (d *DirInfoCache) Load(dir string) (directoryPackageInfo, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()
	info, ok := d.dirs[dir]
	if !ok {
		return directoryPackageInfo{}, false
	}
	return *info, true
}

// Keys returns the keys currently present in d.
func (d *DirInfoCache) Keys() (keys []string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for key := range d.dirs {
		keys = append(keys, key)
	}
	return keys
}

func (d *DirInfoCache) CachePackageName(info directoryPackageInfo) (string, error) {
	if loaded, err := info.reachedStatus(nameLoaded); loaded {
		return info.packageName, err
	}
	if scanned, err := info.reachedStatus(directoryScanned); !scanned || err != nil {
		return "", fmt.Errorf("cannot read package name, scan error: %v", err)
	}
	info.packageName, info.err = packageDirToName(info.dir)
	info.status = nameLoaded
	d.Store(info.dir, info)
	return info.packageName, info.err
}

func (d *DirInfoCache) CacheExports(ctx context.Context, env *ProcessEnv, info directoryPackageInfo) (string, []string, error) {
	if reached, _ := info.reachedStatus(exportsLoaded); reached {
		return info.packageName, info.exports, info.err
	}
	if reached, err := info.reachedStatus(nameLoaded); reached && err != nil {
		return "", nil, err
	}
	info.packageName, info.exports, info.err = loadExportsFromFiles(ctx, env, info.dir, false)
	if info.err == context.Canceled || info.err == context.DeadlineExceeded {
		return info.packageName, info.exports, info.err
	}
	// The cache structure wants things to proceed linearly. We can skip a
	// step here, but only if we succeed.
	if info.status == nameLoaded || info.err == nil {
		info.status = exportsLoaded
	} else {
		info.status = nameLoaded
	}
	d.Store(info.dir, info)
	return info.packageName, info.exports, info.err
}

// ScanModuleCache walks the given directory, which must be a GOMODCACHE value,
// for directory package information, storing the results in cache.
func ScanModuleCache(dir string, cache *DirInfoCache, logf func(string, ...any)) {
	// Note(rfindley): it's hard to see, but this function attempts to implement
	// just the side effects on cache of calling PrimeCache with a ProcessEnv
	// that has the given dir as its GOMODCACHE.
	//
	// Teasing out the control flow, we see that we can avoid any handling of
	// vendor/ and can infer module info entirely from the path, simplifying the
	// logic here.

	root := gopathwalk.Root{
		Path: filepath.Clean(dir),
		Type: gopathwalk.RootModuleCache,
	}

	directoryInfo := func(root gopathwalk.Root, dir string) directoryPackageInfo {
		// This is a copy of ModuleResolver.scanDirForPackage, trimmed down to
		// logic that applies to a module cache directory.

		subdir := ""
		if dir != root.Path {
			subdir = dir[len(root.Path)+len("/"):]
		}

		matches := modCacheRegexp.FindStringSubmatch(subdir)
		if len(matches) == 0 {
			return directoryPackageInfo{
				status: directoryScanned,
				err:    fmt.Errorf("invalid module cache path: %v", subdir),
			}
		}
		modPath, err := module.UnescapePath(filepath.ToSlash(matches[1]))
		if err != nil {
			if logf != nil {
				logf("decoding module cache path %q: %v", subdir, err)
			}
			return directoryPackageInfo{
				status: directoryScanned,
				err:    fmt.Errorf("decoding module cache path %q: %v", subdir, err),
			}
		}
		importPath := path.Join(modPath, filepath.ToSlash(matches[3]))
		index := strings.Index(dir, matches[1]+"@"+matches[2])
		modDir := filepath.Join(dir[:index], matches[1]+"@"+matches[2])
		modName := readModName(filepath.Join(modDir, "go.mod"))
		return directoryPackageInfo{
			status:                 directoryScanned,
			dir:                    dir,
			rootType:               root.Type,
			nonCanonicalImportPath: importPath,
			moduleDir:              modDir,
			moduleName:             modName,
		}
	}

	add := func(root gopathwalk.Root, dir string) {
		info := directoryInfo(root, dir)
		cache.Store(info.dir, info)
	}

	skip := func(_ gopathwalk.Root, dir string) bool {
		// Skip directories that have already been scanned.
		//
		// Note that gopathwalk only adds "package" directories, which must contain
		// a .go file, and all such package directories in the module cache are
		// immutable. So if we can load a dir, it can be skipped.
		info, ok := cache.Load(dir)
		if !ok {
			return false
		}
		packageScanned, _ := info.reachedStatus(directoryScanned)
		return packageScanned
	}

	gopathwalk.WalkSkip([]gopathwalk.Root{root}, add, skip, gopathwalk.Options{Logf: logf, ModulesEnabled: true})
}
