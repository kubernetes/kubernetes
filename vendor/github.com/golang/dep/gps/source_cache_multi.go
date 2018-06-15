// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"github.com/golang/dep/gps/pkgtree"
)

// A multiCache manages two cache levels, ephemeral in-memory and persistent on-disk.
//
// The in-memory cache is always checked first, with the on-disk used as a fallback.
// Values read from disk are set in-memory when an appropriate method exists.
//
// Set values are cached both in-memory and on-disk.
type multiCache struct {
	mem, disk singleSourceCache
}

func (c *multiCache) setManifestAndLock(r Revision, ai ProjectAnalyzerInfo, m Manifest, l Lock) {
	c.mem.setManifestAndLock(r, ai, m, l)
	c.disk.setManifestAndLock(r, ai, m, l)
}

func (c *multiCache) getManifestAndLock(r Revision, ai ProjectAnalyzerInfo) (Manifest, Lock, bool) {
	m, l, ok := c.mem.getManifestAndLock(r, ai)
	if ok {
		return m, l, true
	}

	m, l, ok = c.disk.getManifestAndLock(r, ai)
	if ok {
		c.mem.setManifestAndLock(r, ai, m, l)
		return m, l, true
	}

	return nil, nil, false
}

func (c *multiCache) setPackageTree(r Revision, ptree pkgtree.PackageTree) {
	c.mem.setPackageTree(r, ptree)
	c.disk.setPackageTree(r, ptree)
}

func (c *multiCache) getPackageTree(r Revision) (pkgtree.PackageTree, bool) {
	ptree, ok := c.mem.getPackageTree(r)
	if ok {
		return ptree, true
	}

	ptree, ok = c.disk.getPackageTree(r)
	if ok {
		c.mem.setPackageTree(r, ptree)
		return ptree, true
	}

	return pkgtree.PackageTree{}, false
}

func (c *multiCache) markRevisionExists(r Revision) {
	c.mem.markRevisionExists(r)
	c.disk.markRevisionExists(r)
}

func (c *multiCache) setVersionMap(pvs []PairedVersion) {
	c.mem.setVersionMap(pvs)
	c.disk.setVersionMap(pvs)
}

func (c *multiCache) getVersionsFor(rev Revision) ([]UnpairedVersion, bool) {
	uvs, ok := c.mem.getVersionsFor(rev)
	if ok {
		return uvs, true
	}

	return c.disk.getVersionsFor(rev)
}

func (c *multiCache) getAllVersions() ([]PairedVersion, bool) {
	pvs, ok := c.mem.getAllVersions()
	if ok {
		return pvs, true
	}

	pvs, ok = c.disk.getAllVersions()
	if ok {
		c.mem.setVersionMap(pvs)
		return pvs, true
	}

	return nil, false
}

func (c *multiCache) getRevisionFor(uv UnpairedVersion) (Revision, bool) {
	rev, ok := c.mem.getRevisionFor(uv)
	if ok {
		return rev, true
	}

	return c.disk.getRevisionFor(uv)
}

func (c *multiCache) toRevision(v Version) (Revision, bool) {
	rev, ok := c.mem.toRevision(v)
	if ok {
		return rev, true
	}

	return c.disk.toRevision(v)
}

func (c *multiCache) toUnpaired(v Version) (UnpairedVersion, bool) {
	uv, ok := c.mem.toUnpaired(v)
	if ok {
		return uv, true
	}

	return c.disk.toUnpaired(v)
}
