// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/golang/dep"
	"github.com/golang/dep/gps"
	"github.com/golang/dep/gps/paths"
	"github.com/golang/dep/gps/pkgtree"
	fb "github.com/golang/dep/internal/feedback"
	"github.com/golang/dep/internal/fs"
	"github.com/golang/dep/internal/kdep"
	"github.com/pkg/errors"
)

// gopathScanner supplies manifest/lock data by scanning the contents of GOPATH
// It uses its results to fill-in any missing details left by the rootAnalyzer.
type gopathScanner struct {
	ctx        *kdep.Ctx
	directDeps map[gps.ProjectRoot]bool
	sm         gps.SourceManager

	pd    projectData
	origM *dep.Manifest
	origL *dep.Lock
}

func newGopathScanner(ctx *kdep.Ctx, directDeps map[gps.ProjectRoot]bool, sm gps.SourceManager) *gopathScanner {
	return &gopathScanner{
		ctx:        ctx,
		directDeps: directDeps,
		sm:         sm,
	}
}

// InitializeRootManifestAndLock performs analysis of the filesystem tree rooted
// at path, with the root import path importRoot, to determine the project's
// constraints. Respect any initial constraints defined in the root manifest and
// lock.
func (g *gopathScanner) InitializeRootManifestAndLock(rootM *dep.Manifest, rootL *dep.Lock) error {
	var err error

	g.ctx.Err.Println("Searching GOPATH for projects...")
	g.pd, err = g.scanGopathForDependencies()
	if err != nil {
		return err
	}

	g.origM = dep.NewManifest()
	g.origM.Constraints = g.pd.constraints

	g.origL = &dep.Lock{
		P: make([]gps.LockedProject, 0, len(g.pd.ondisk)),
	}

	for pr, v := range g.pd.ondisk {
		// That we have to chop off these path prefixes is a symptom of
		// a problem in gps itself
		pkgs := make([]string, 0, len(g.pd.dependencies[pr]))
		prslash := string(pr) + "/"
		for _, pkg := range g.pd.dependencies[pr] {
			if pkg == string(pr) {
				pkgs = append(pkgs, ".")
			} else {
				pkgs = append(pkgs, trimPathPrefix(pkg, prslash))
			}
		}

		g.origL.P = append(g.origL.P, gps.NewLockedProject(
			gps.ProjectIdentifier{ProjectRoot: pr}, v, pkgs),
		)
	}

	g.overlay(rootM, rootL)

	return nil
}

// Fill in gaps in the root manifest/lock with data found from the GOPATH.
func (g *gopathScanner) overlay(rootM *dep.Manifest, rootL *dep.Lock) {
	for pkg, prj := range g.origM.Constraints {
		if _, has := rootM.Constraints[pkg]; has {
			continue
		}
		rootM.Constraints[pkg] = prj
		v := g.pd.ondisk[pkg]

		pi := gps.ProjectIdentifier{ProjectRoot: pkg, Source: prj.Source}
		f := fb.NewConstraintFeedback(gps.ProjectConstraint{Ident: pi, Constraint: v}, fb.DepTypeDirect)
		f.LogFeedback(g.ctx.Err)
		f = fb.NewLockedProjectFeedback(gps.NewLockedProject(pi, v, nil), fb.DepTypeDirect)
		f.LogFeedback(g.ctx.Err)
	}

	// Keep track of which projects have been locked
	lockedProjects := map[gps.ProjectRoot]bool{}
	for _, lp := range rootL.P {
		lockedProjects[lp.Ident().ProjectRoot] = true
	}

	for _, lp := range g.origL.P {
		pkg := lp.Ident().ProjectRoot
		if _, isLocked := lockedProjects[pkg]; isLocked {
			continue
		}
		rootL.P = append(rootL.P, lp)
		lockedProjects[pkg] = true

		if _, isDirect := g.directDeps[pkg]; !isDirect {
			f := fb.NewLockedProjectFeedback(lp, fb.DepTypeTransitive)
			f.LogFeedback(g.ctx.Err)
		}
	}

	// Identify projects whose version is unknown and will have to be solved for
	var missing []string    // all project roots missing from GOPATH
	var missingVCS []string // all project roots missing VCS information
	for pr := range g.pd.notondisk {
		if _, isLocked := lockedProjects[pr]; isLocked {
			continue
		}
		if g.pd.invalidSVC[pr] {
			missingVCS = append(missingVCS, string(pr))
		} else {
			missing = append(missing, string(pr))
		}
	}

	missingStr := ""
	missingVCSStr := ""
	if len(missing) > 0 {
		missingStr = fmt.Sprintf("The following dependencies were not found in GOPATH:\n  %s\n\n",
			strings.Join(missing, "\n  "))
	}
	if len(missingVCS) > 0 {
		missingVCSStr = fmt.Sprintf("The following dependencies found in GOPATH were missing VCS information (a remote source is required):\n  %s\n\n",
			strings.Join(missingVCS, "\n  "))
	}
	if len(missingVCS)+len(missing) > 0 {
		g.ctx.Err.Printf("\n%s%sThe most recent version of these projects will be used.\n\n", missingStr, missingVCSStr)
	}
}

func trimPathPrefix(p1, p2 string) string {
	if isPrefix, _ := fs.HasFilepathPrefix(p1, p2); isPrefix {
		return p1[len(p2):]
	}
	return p1
}

// contains checks if a array of strings contains a value
func contains(a []string, b string) bool {
	for _, v := range a {
		if b == v {
			return true
		}
	}
	return false
}

// getProjectPropertiesFromVersion takes a Version and returns a proper
// ProjectProperties with Constraint value based on the provided version.
func getProjectPropertiesFromVersion(v gps.Version) gps.ProjectProperties {
	pp := gps.ProjectProperties{}

	// extract version and ignore if it's revision only
	switch tv := v.(type) {
	case gps.PairedVersion:
		v = tv.Unpair()
	case gps.Revision:
		return pp
	}

	switch v.Type() {
	case gps.IsBranch, gps.IsVersion:
		pp.Constraint = v
	case gps.IsSemver:
		c, err := gps.NewSemverConstraintIC(v.String())
		if err != nil {
			panic(err)
		}
		pp.Constraint = c
	}

	return pp
}

type projectData struct {
	constraints  gps.ProjectConstraints          // constraints that could be found
	dependencies map[gps.ProjectRoot][]string    // all dependencies (imports) found by project root
	notondisk    map[gps.ProjectRoot]bool        // projects that were not found on disk
	invalidSVC   map[gps.ProjectRoot]bool        // projects that were found on disk but SVC data could not be read
	ondisk       map[gps.ProjectRoot]gps.Version // projects that were found on disk
}

func (g *gopathScanner) scanGopathForDependencies() (projectData, error) {
	constraints := make(gps.ProjectConstraints)
	dependencies := make(map[gps.ProjectRoot][]string)
	packages := make(map[string]bool)
	notondisk := make(map[gps.ProjectRoot]bool)
	invalidSVC := make(map[gps.ProjectRoot]bool)
	ondisk := make(map[gps.ProjectRoot]gps.Version)

	var syncDepGroup sync.WaitGroup
	syncDep := func(pr gps.ProjectRoot, sm gps.SourceManager) {
		if err := sm.SyncSourceFor(gps.ProjectIdentifier{ProjectRoot: pr}); err != nil {
			g.ctx.Err.Printf("%+v", errors.Wrapf(err, "Unable to cache %s", pr))
		}
		syncDepGroup.Done()
	}

	if len(g.directDeps) == 0 {
		return projectData{}, nil
	}

	for ippr := range g.directDeps {
		// TODO(sdboyer) these are not import paths by this point, they've
		// already been worked down to project roots.
		ip := string(ippr)
		pr, err := g.sm.DeduceProjectRoot(ip)
		if err != nil {
			return projectData{}, errors.Wrap(err, "sm.DeduceProjectRoot")
		}

		packages[ip] = true
		if _, has := dependencies[pr]; has {
			dependencies[pr] = append(dependencies[pr], ip)
			continue
		}
		syncDepGroup.Add(1)
		go syncDep(pr, g.sm)

		dependencies[pr] = []string{ip}
		abs, err := g.ctx.AbsForImport(string(pr))
		if err != nil {
			notondisk[pr] = true
			continue
		}
		v, err := gps.VCSVersion(abs)
		if err != nil {
			invalidSVC[pr] = true
			notondisk[pr] = true
			continue
		}

		ondisk[pr] = v
		pp := getProjectPropertiesFromVersion(v)
		if pp.Constraint != nil || pp.Source != "" {
			constraints[pr] = pp
		}
	}

	// Explore the packages we've found for transitive deps, either
	// completing the lock or identifying (more) missing projects that we'll
	// need to ask gps to solve for us.
	colors := make(map[string]uint8)
	const (
		white uint8 = iota
		grey
		black
	)

	// cache of PackageTrees, so we don't parse projects more than once
	ptrees := make(map[gps.ProjectRoot]pkgtree.PackageTree)

	// depth-first traverser
	var dft func(string) error
	dft = func(pkg string) error {
		switch colors[pkg] {
		case white:
			colors[pkg] = grey

			pr, err := g.sm.DeduceProjectRoot(pkg)
			if err != nil {
				return errors.Wrap(err, "could not deduce project root for "+pkg)
			}

			// We already visited this project root earlier via some other
			// pkg within it, and made the decision that it's not on disk.
			// Respect that decision, and pop the stack.
			if notondisk[pr] {
				colors[pkg] = black
				return nil
			}

			ptree, has := ptrees[pr]
			if !has {
				// It's fine if the root does not exist - it indicates that this
				// project is not present in the workspace, and so we need to
				// solve to deal with this dep.
				r := filepath.Join(g.ctx.GOPATH, "src", string(pr))
				fi, err := os.Stat(r)
				if os.IsNotExist(err) || !fi.IsDir() {
					colors[pkg] = black
					notondisk[pr] = true
					return nil
				}

				// We know the project is on disk; the question is whether we're
				// first seeing it here, in the transitive exploration, or if it
				// was found in the initial pass on direct imports. We know it's
				// the former if there's no entry for it in the ondisk map.
				if _, in := ondisk[pr]; !in {
					abs, err := g.ctx.AbsForImport(string(pr))
					if err != nil {
						colors[pkg] = black
						notondisk[pr] = true
						return nil
					}
					v, err := gps.VCSVersion(abs)
					if err != nil {
						// Even if we know it's on disk, errors are still
						// possible when trying to deduce version. If we
						// encounter such an error, just treat the project as
						// not being on disk; the solver will work it out.
						colors[pkg] = black
						notondisk[pr] = true
						return nil
					}
					ondisk[pr] = v
				}

				ptree, err = pkgtree.ListPackages(r, string(pr))
				if err != nil {
					// Any error here other than an a nonexistent dir (which
					// can't happen because we covered that case above) is
					// probably critical, so bail out.
					return errors.Wrap(err, "gps.ListPackages")
				}
				ptrees[pr] = ptree
			}

			// Get a reachmap that includes main pkgs (even though importing
			// them is an error, what we're checking right now is simply whether
			// there's a package with go code present on disk), and does not
			// backpropagate errors (again, because our only concern right now
			// is package existence).
			rm, errmap := ptree.ToReachMap(true, false, false, nil)
			reached, ok := rm[pkg]
			if !ok {
				colors[pkg] = black
				// not on disk...
				notondisk[pr] = true
				return nil
			}
			if _, ok := errmap[pkg]; ok {
				// The package is on disk, but contains some errors.
				colors[pkg] = black
				return nil
			}

			if deps, has := dependencies[pr]; has {
				if !contains(deps, pkg) {
					dependencies[pr] = append(deps, pkg)
				}
			} else {
				dependencies[pr] = []string{pkg}
				syncDepGroup.Add(1)
				go syncDep(pr, g.sm)
			}

			// recurse
			for _, rpkg := range reached.External {
				if paths.IsStandardImportPath(rpkg) {
					continue
				}

				err := dft(rpkg)
				if err != nil {
					// Bubble up any errors we encounter
					return err
				}
			}

			colors[pkg] = black
		case grey:
			return errors.Errorf("Import cycle detected on %s", pkg)
		}
		return nil
	}

	// run the depth-first traversal from the set of immediate external
	// package imports we found in the current project
	for pkg := range packages {
		err := dft(pkg)
		if err != nil {
			return projectData{}, err // already errors.Wrap()'d internally
		}
	}

	syncDepGroup.Wait()

	pd := projectData{
		constraints:  constraints,
		dependencies: dependencies,
		invalidSVC:   invalidSVC,
		notondisk:    notondisk,
		ondisk:       ondisk,
	}
	return pd, nil
}
