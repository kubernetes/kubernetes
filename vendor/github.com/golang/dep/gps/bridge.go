// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"

	"github.com/golang/dep/gps/pkgtree"
)

// sourceBridge is an adapter to SourceManagers that tailor operations for a
// single solve run.
type sourceBridge interface {
	// sourceBridge includes all the methods in the SourceManager interface except
	// for Release().
	SourceExists(ProjectIdentifier) (bool, error)
	SyncSourceFor(ProjectIdentifier) error
	RevisionPresentIn(ProjectIdentifier, Revision) (bool, error)
	ListPackages(ProjectIdentifier, Version) (pkgtree.PackageTree, error)
	GetManifestAndLock(ProjectIdentifier, Version, ProjectAnalyzer) (Manifest, Lock, error)
	ExportProject(ProjectIdentifier, Version, string) error
	DeduceProjectRoot(ip string) (ProjectRoot, error)

	//sourceExists(ProjectIdentifier) (bool, error)
	//syncSourceFor(ProjectIdentifier) error
	listVersions(ProjectIdentifier) ([]Version, error)
	//revisionPresentIn(ProjectIdentifier, Revision) (bool, error)
	//listPackages(ProjectIdentifier, Version) (pkgtree.PackageTree, error)
	//getManifestAndLock(ProjectIdentifier, Version, ProjectAnalyzer) (Manifest, Lock, error)
	//exportProject(ProjectIdentifier, Version, string) error
	//deduceProjectRoot(ip string) (ProjectRoot, error)
	verifyRootDir(path string) error
	vendorCodeExists(ProjectIdentifier) (bool, error)
	breakLock()
}

// bridge is an adapter around a proper SourceManager. It provides localized
// caching that's tailored to the requirements of a particular solve run.
//
// Finally, it provides authoritative version/constraint operations, ensuring
// that any possible approach to a match - even those not literally encoded in
// the inputs - is achieved.
type bridge struct {
	// The underlying, adapted-to SourceManager
	sm SourceManager

	// The solver which we're assisting.
	//
	// The link between solver and bridge is circular, which is typically a bit
	// awkward, but the bridge needs access to so many of the input arguments
	// held by the solver that it ends up being easier and saner to do this.
	s *solver

	// Map of project root name to their available version list. This cache is
	// layered on top of the proper SourceManager's cache; the only difference
	// is that this keeps the versions sorted in the direction required by the
	// current solve run.
	vlists map[ProjectIdentifier][]Version

	// Indicates whether lock breaking has already been run
	lockbroken int32

	// Whether to sort version lists for downgrade.
	down bool

	// The cancellation context provided to the solver. Threading it through the
	// various solver methods is needlessly verbose so long as we maintain the
	// lifetime guarantees that a solver can only be run once.
	// TODO(sdboyer) uncomment this and thread it through SourceManager methods
	//ctx context.Context
}

// mkBridge creates a bridge
func mkBridge(s *solver, sm SourceManager, down bool) *bridge {
	return &bridge{
		sm:     sm,
		s:      s,
		down:   down,
		vlists: make(map[ProjectIdentifier][]Version),
	}
}

func (b *bridge) GetManifestAndLock(id ProjectIdentifier, v Version, an ProjectAnalyzer) (Manifest, Lock, error) {
	if b.s.rd.isRoot(id.ProjectRoot) {
		return b.s.rd.rm, b.s.rd.rl, nil
	}

	b.s.mtr.push("b-gmal")
	m, l, e := b.sm.GetManifestAndLock(id, v, an)
	b.s.mtr.pop()
	return m, l, e
}

func (b *bridge) listVersions(id ProjectIdentifier) ([]Version, error) {
	if vl, exists := b.vlists[id]; exists {
		return vl, nil
	}

	b.s.mtr.push("b-list-versions")
	pvl, err := b.sm.ListVersions(id)
	if err != nil {
		b.s.mtr.pop()
		return nil, err
	}

	vl := hidePair(pvl)
	if b.down {
		SortForDowngrade(vl)
	} else {
		SortForUpgrade(vl)
	}

	b.vlists[id] = vl
	b.s.mtr.pop()
	return vl, nil
}

func (b *bridge) RevisionPresentIn(id ProjectIdentifier, r Revision) (bool, error) {
	b.s.mtr.push("b-rev-present-in")
	i, e := b.sm.RevisionPresentIn(id, r)
	b.s.mtr.pop()
	return i, e
}

func (b *bridge) SourceExists(id ProjectIdentifier) (bool, error) {
	b.s.mtr.push("b-source-exists")
	i, e := b.sm.SourceExists(id)
	b.s.mtr.pop()
	return i, e
}

func (b *bridge) vendorCodeExists(id ProjectIdentifier) (bool, error) {
	fi, err := os.Stat(filepath.Join(b.s.rd.dir, "vendor", string(id.ProjectRoot)))
	if err != nil {
		return false, err
	} else if fi.IsDir() {
		return true, nil
	}

	return false, nil
}

// listPackages lists all the packages contained within the given project at a
// particular version.
//
// The root project is handled separately, as the source manager isn't
// responsible for that code.
func (b *bridge) ListPackages(id ProjectIdentifier, v Version) (pkgtree.PackageTree, error) {
	if b.s.rd.isRoot(id.ProjectRoot) {
		return b.s.rd.rpt, nil
	}

	b.s.mtr.push("b-list-pkgs")
	pt, err := b.sm.ListPackages(id, v)
	b.s.mtr.pop()
	return pt, err
}

func (b *bridge) ExportProject(id ProjectIdentifier, v Version, path string) error {
	panic("bridge should never be used to ExportProject")
}

// verifyRoot ensures that the provided path to the project root is in good
// working condition. This check is made only once, at the beginning of a solve
// run.
func (b *bridge) verifyRootDir(path string) error {
	if fi, err := os.Stat(path); err != nil {
		return badOptsFailure(fmt.Sprintf("could not read project root (%s): %s", path, err))
	} else if !fi.IsDir() {
		return badOptsFailure(fmt.Sprintf("project root (%s) is a file, not a directory", path))
	}

	return nil
}

func (b *bridge) DeduceProjectRoot(ip string) (ProjectRoot, error) {
	b.s.mtr.push("b-deduce-proj-root")
	pr, e := b.sm.DeduceProjectRoot(ip)
	b.s.mtr.pop()
	return pr, e
}

// breakLock is called when the solver has to break a version recorded in the
// lock file. It prefetches all the projects in the solver's lock, so that the
// information is already on hand if/when the solver needs it.
//
// Projects that have already been selected are skipped, as it's generally unlikely that the
// solver will have to backtrack through and fully populate their version queues.
func (b *bridge) breakLock() {
	// No real conceivable circumstance in which multiple calls are made to
	// this, but being that this is the entrance point to a bunch of async work,
	// protect it with an atomic CAS in case things change in the future.
	//
	// We avoid using a sync.Once here, as there's no reason for other callers
	// to block until completion.
	if !atomic.CompareAndSwapInt32(&b.lockbroken, 0, 1) {
		return
	}

	for _, lp := range b.s.rd.rl.Projects() {
		if _, is := b.s.sel.selected(lp.pi); !is {
			pi, v := lp.pi, lp.Version()
			go func() {
				// Sync first
				b.sm.SyncSourceFor(pi)
				// Preload the package info for the locked version, too, as
				// we're more likely to need that
				b.sm.ListPackages(pi, v)
			}()
		}
	}
}

func (b *bridge) SyncSourceFor(id ProjectIdentifier) error {
	// we don't track metrics here b/c this is often called in its own goroutine
	// by the solver, and the metrics design is for wall time on a single thread
	return b.sm.SyncSourceFor(id)
}
