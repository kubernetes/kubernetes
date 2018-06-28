// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"fmt"
	"sync"

	"github.com/golang/dep/gps/pkgtree"
)

// singleSourceCache provides a method set for storing and retrieving data about
// a single source.
type singleSourceCache interface {
	// Store the manifest and lock information for a given revision, as defined by
	// a particular ProjectAnalyzer.
	setManifestAndLock(Revision, ProjectAnalyzerInfo, Manifest, Lock)

	// Get the manifest and lock information for a given revision, as defined by
	// a particular ProjectAnalyzer.
	getManifestAndLock(Revision, ProjectAnalyzerInfo) (Manifest, Lock, bool)

	// Store a PackageTree for a given revision.
	setPackageTree(Revision, pkgtree.PackageTree)

	// Get the PackageTree for a given revision.
	getPackageTree(Revision) (pkgtree.PackageTree, bool)

	// Indicate to the cache that an individual revision is known to exist.
	markRevisionExists(r Revision)

	// Store the mappings between a set of PairedVersions' surface versions
	// their corresponding revisions.
	//
	// The existing list of versions will be purged before writing. Revisions
	// will have their pairings purged, but record of the revision existing will
	// be kept, on the assumption that revisions are immutable and permanent.
	setVersionMap(versionList []PairedVersion)

	// Get the list of unpaired versions corresponding to the given revision.
	getVersionsFor(Revision) ([]UnpairedVersion, bool)

	// Gets all the version pairs currently known to the cache.
	getAllVersions() ([]PairedVersion, bool)

	// Get the revision corresponding to the given unpaired version.
	getRevisionFor(UnpairedVersion) (Revision, bool)

	// Attempt to convert the given Version to a Revision, given information
	// currently present in the cache, and in the Version itself.
	toRevision(v Version) (Revision, bool)

	// Attempt to convert the given Version to an UnpairedVersion, given
	// information currently present in the cache, or in the Version itself.
	//
	// If the input is a revision and multiple UnpairedVersions are associated
	// with it, whatever happens to be the first is returned.
	toUnpaired(v Version) (UnpairedVersion, bool)
}

type singleSourceCacheMemory struct {
	mut    sync.RWMutex // protects all fields
	infos  map[ProjectAnalyzerInfo]map[Revision]projectInfo
	ptrees map[Revision]pkgtree.PackageTree
	vList  []PairedVersion // replaced, never modified
	vMap   map[UnpairedVersion]Revision
	rMap   map[Revision][]UnpairedVersion
}

func newMemoryCache() singleSourceCache {
	return &singleSourceCacheMemory{
		infos:  make(map[ProjectAnalyzerInfo]map[Revision]projectInfo),
		ptrees: make(map[Revision]pkgtree.PackageTree),
		vMap:   make(map[UnpairedVersion]Revision),
		rMap:   make(map[Revision][]UnpairedVersion),
	}
}

type projectInfo struct {
	Manifest
	Lock
}

func (c *singleSourceCacheMemory) setManifestAndLock(r Revision, pai ProjectAnalyzerInfo, m Manifest, l Lock) {
	c.mut.Lock()
	inner, has := c.infos[pai]
	if !has {
		inner = make(map[Revision]projectInfo)
		c.infos[pai] = inner
	}
	inner[r] = projectInfo{Manifest: m, Lock: l}

	// Ensure there's at least an entry in the rMap so that the rMap always has
	// a complete picture of the revisions we know to exist
	if _, has = c.rMap[r]; !has {
		c.rMap[r] = nil
	}
	c.mut.Unlock()
}

func (c *singleSourceCacheMemory) getManifestAndLock(r Revision, pai ProjectAnalyzerInfo) (Manifest, Lock, bool) {
	c.mut.Lock()
	defer c.mut.Unlock()

	inner, has := c.infos[pai]
	if !has {
		return nil, nil, false
	}

	pi, has := inner[r]
	if has {
		return pi.Manifest, pi.Lock, true
	}
	return nil, nil, false
}

func (c *singleSourceCacheMemory) setPackageTree(r Revision, ptree pkgtree.PackageTree) {
	c.mut.Lock()
	c.ptrees[r] = ptree

	// Ensure there's at least an entry in the rMap so that the rMap always has
	// a complete picture of the revisions we know to exist
	if _, has := c.rMap[r]; !has {
		c.rMap[r] = nil
	}
	c.mut.Unlock()
}

func (c *singleSourceCacheMemory) getPackageTree(r Revision) (pkgtree.PackageTree, bool) {
	c.mut.Lock()
	ptree, has := c.ptrees[r]
	c.mut.Unlock()
	return ptree, has
}

func (c *singleSourceCacheMemory) setVersionMap(versionList []PairedVersion) {
	c.mut.Lock()
	c.vList = versionList
	// TODO(sdboyer) how do we handle cache consistency here - revs that may
	// be out of date vis-a-vis the ptrees or infos maps?
	for r := range c.rMap {
		c.rMap[r] = nil
	}

	c.vMap = make(map[UnpairedVersion]Revision, len(versionList))

	for _, pv := range versionList {
		u, r := pv.Unpair(), pv.Revision()
		c.vMap[u] = r
		c.rMap[r] = append(c.rMap[r], u)
	}
	c.mut.Unlock()
}

func (c *singleSourceCacheMemory) markRevisionExists(r Revision) {
	c.mut.Lock()
	if _, has := c.rMap[r]; !has {
		c.rMap[r] = nil
	}
	c.mut.Unlock()
}

func (c *singleSourceCacheMemory) getVersionsFor(r Revision) ([]UnpairedVersion, bool) {
	c.mut.Lock()
	versionList, has := c.rMap[r]
	c.mut.Unlock()
	return versionList, has
}

func (c *singleSourceCacheMemory) getAllVersions() ([]PairedVersion, bool) {
	c.mut.Lock()
	vList := c.vList
	c.mut.Unlock()

	if vList == nil {
		return nil, false
	}
	cp := make([]PairedVersion, len(vList))
	copy(cp, vList)
	return cp, true
}

func (c *singleSourceCacheMemory) getRevisionFor(uv UnpairedVersion) (Revision, bool) {
	c.mut.Lock()
	r, has := c.vMap[uv]
	c.mut.Unlock()
	return r, has
}

func (c *singleSourceCacheMemory) toRevision(v Version) (Revision, bool) {
	switch t := v.(type) {
	case Revision:
		return t, true
	case PairedVersion:
		return t.Revision(), true
	case UnpairedVersion:
		c.mut.Lock()
		r, has := c.vMap[t]
		c.mut.Unlock()
		return r, has
	default:
		panic(fmt.Sprintf("Unknown version type %T", v))
	}
}

func (c *singleSourceCacheMemory) toUnpaired(v Version) (UnpairedVersion, bool) {
	switch t := v.(type) {
	case UnpairedVersion:
		return t, true
	case PairedVersion:
		return t.Unpair(), true
	case Revision:
		c.mut.Lock()
		upv, has := c.rMap[t]
		c.mut.Unlock()

		if has && len(upv) > 0 {
			return upv[0], true
		}
		return nil, false
	default:
		panic(fmt.Sprintf("unknown version type %T", v))
	}
}
