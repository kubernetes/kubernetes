// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"log"
	"strings"

	"github.com/golang/dep"
	"github.com/golang/dep/gps"
	fb "github.com/golang/dep/internal/feedback"
	"github.com/pkg/errors"
)

// Importer provides a common implementation for importing from other
// dependency managers.
type Importer struct {
	SourceManager gps.SourceManager
	Logger        *log.Logger
	Verbose       bool
	Manifest      *dep.Manifest
	Lock          *dep.Lock
}

// NewImporter creates a new Importer for embedding in an importer.
func NewImporter(logger *log.Logger, verbose bool, sm gps.SourceManager) *Importer {
	return &Importer{
		Logger:        logger,
		Verbose:       verbose,
		Manifest:      dep.NewManifest(),
		Lock:          &dep.Lock{},
		SourceManager: sm,
	}
}

// isTag determines if the specified value is a tag (plain or semver).
func (i *Importer) isTag(pi gps.ProjectIdentifier, value string) (bool, gps.Version, error) {
	versions, err := i.SourceManager.ListVersions(pi)
	if err != nil {
		return false, nil, errors.Wrapf(err, "unable to list versions for %s(%s)", pi.ProjectRoot, pi.Source)
	}

	for _, version := range versions {
		if version.Type() != gps.IsVersion && version.Type() != gps.IsSemver {
			continue
		}

		if value == version.String() {
			return true, version, nil
		}
	}

	return false, nil, nil
}

// lookupVersionForLockedProject figures out the appropriate version for a locked
// project based on the locked revision and the constraint from the manifest.
// First try matching the revision to a version, then try the constraint from the
// manifest, then finally the revision.
func (i *Importer) lookupVersionForLockedProject(pi gps.ProjectIdentifier, c gps.Constraint, rev gps.Revision) (gps.Version, error) {
	// Find the version that goes with this revision, if any
	versions, err := i.SourceManager.ListVersions(pi)
	if err != nil {
		return rev, errors.Wrapf(err, "Unable to lookup the version represented by %s in %s(%s). Falling back to locking the revision only.", rev, pi.ProjectRoot, pi.Source)
	}

	var branchConstraint gps.PairedVersion
	gps.SortPairedForUpgrade(versions) // Sort versions in asc order
	matches := []gps.Version{}
	for _, v := range versions {
		if v.Revision() == rev {
			matches = append(matches, v)
		}
		if c != nil && v.Type() == gps.IsBranch && v.String() == c.String() {
			branchConstraint = v
		}
	}

	// Try to narrow down the matches with the constraint. Otherwise return the first match.
	if len(matches) > 0 {
		if c != nil {
			for _, v := range matches {
				if i.testConstraint(c, v) {
					return v, nil
				}
			}
		}
		return matches[0], nil
	}

	// Use branch constraint from the manifest
	if branchConstraint != nil {
		return branchConstraint.Unpair().Pair(rev), nil
	}

	// Give up and lock only to a revision
	return rev, nil
}

// ImportedPackage is a common intermediate representation of a package imported
// from an external tool's configuration.
type ImportedPackage struct {
	// Required. The package path, not necessarily the project root.
	Name string

	// Required. Text representing a revision or tag.
	LockHint string

	// Optional. Alternative source, or fork, for the project.
	Source string

	// Optional. Text representing a branch or version.
	ConstraintHint string
}

// importedProject is a consolidated representation of a set of imported packages
// for the same project root.
type importedProject struct {
	Root gps.ProjectRoot
	ImportedPackage
}

// loadPackages consolidates all package references into a set of project roots.
func (i *Importer) loadPackages(packages []ImportedPackage) []importedProject {
	// preserve the original order of the packages so that messages that
	// are printed as they are processed are in a consistent order.
	orderedProjects := make([]importedProject, 0, len(packages))

	projects := make(map[gps.ProjectRoot]*importedProject, len(packages))
	for _, pkg := range packages {
		pr, err := i.SourceManager.DeduceProjectRoot(pkg.Name)
		if err != nil {
			i.Logger.Printf(
				"  Warning: Skipping project. Cannot determine the project root for %s: %s\n",
				pkg.Name, err,
			)
			continue
		}
		pkg.Name = string(pr)

		prj, exists := projects[pr]
		if !exists {
			prj := importedProject{pr, pkg}
			orderedProjects = append(orderedProjects, prj)
			projects[pr] = &orderedProjects[len(orderedProjects)-1]
			continue
		}

		// The config found first "wins", though we allow for incrementally
		// setting each field because some importers have a config and lock file.
		if prj.Source == "" && pkg.Source != "" {
			prj.Source = pkg.Source
		}

		if prj.ConstraintHint == "" && pkg.ConstraintHint != "" {
			prj.ConstraintHint = pkg.ConstraintHint
		}

		if prj.LockHint == "" && pkg.LockHint != "" {
			prj.LockHint = pkg.LockHint
		}
	}

	return orderedProjects
}

// ImportPackages loads imported packages into the manifest and lock.
// - defaultConstraintFromLock specifies if a constraint should be defaulted
//   based on the locked version when there wasn't a constraint hint.
//
// Rules:
// * When a constraint is ignored, default to *.
// * HEAD revisions default to the matching branch.
// * Semantic versions default to ^VERSION.
// * Revision constraints are ignored.
// * Versions that don't satisfy the constraint, drop the constraint.
// * Untagged revisions ignore non-branch constraint hints.
func (i *Importer) ImportPackages(packages []ImportedPackage, defaultConstraintFromLock bool) {
	projects := i.loadPackages(packages)

	for _, prj := range projects {
		source := prj.Source
		if len(source) > 0 {
			isDefault, err := i.isDefaultSource(prj.Root, source)
			if err != nil {
				i.Logger.Printf("  Ignoring imported source %s for %s: %s", source, prj.Root, err.Error())
				source = ""
			} else if isDefault {
				source = ""
			} else if strings.Contains(source, "/vendor/") {
				i.Logger.Printf("  Ignoring imported source %s for %s because vendored sources aren't supported", source, prj.Root)
				source = ""
			}
		}

		pc := gps.ProjectConstraint{
			Ident: gps.ProjectIdentifier{
				ProjectRoot: prj.Root,
				Source:      source,
			},
		}

		var err error
		pc.Constraint, err = i.SourceManager.InferConstraint(prj.ConstraintHint, pc.Ident)
		if err != nil {
			pc.Constraint = gps.Any()
		}

		var version gps.Version
		if prj.LockHint != "" {
			var isTag bool
			// Determine if the lock hint is a revision or tag
			isTag, version, err = i.isTag(pc.Ident, prj.LockHint)
			if err != nil {
				i.Logger.Printf(
					"  Warning: Skipping project. Unable to import lock %q for %v: %s\n",
					prj.LockHint, pc.Ident, err,
				)
				continue
			}
			// If the hint is a revision, check if it is tagged
			if !isTag {
				revision := gps.Revision(prj.LockHint)
				version, err = i.lookupVersionForLockedProject(pc.Ident, pc.Constraint, revision)
				if err != nil {
					version = nil
					i.Logger.Println(err)
				}
			}

			// Default the constraint based on the locked version
			if defaultConstraintFromLock && prj.ConstraintHint == "" && version != nil {
				c := i.convertToConstraint(version)
				if c != nil {
					pc.Constraint = c
				}
			}
		}

		// Ignore pinned constraints
		if i.isConstraintPinned(pc.Constraint) {
			if i.Verbose {
				i.Logger.Printf("  Ignoring pinned constraint %v for %v.\n", pc.Constraint, pc.Ident)
			}
			pc.Constraint = gps.Any()
		}

		// Ignore constraints which conflict with the locked revision, so that
		// solve doesn't later change the revision to satisfy the constraint.
		if !i.testConstraint(pc.Constraint, version) {
			if i.Verbose {
				i.Logger.Printf("  Ignoring constraint %v for %v because it would invalidate the locked version %v.\n", pc.Constraint, pc.Ident, version)
			}
			pc.Constraint = gps.Any()
		}

		// Add constraint to manifest that is not empty (has a branch, version or source)
		if !gps.IsAny(pc.Constraint) || pc.Ident.Source != "" {
			i.Manifest.Constraints[pc.Ident.ProjectRoot] = gps.ProjectProperties{
				Source:     pc.Ident.Source,
				Constraint: pc.Constraint,
			}
			fb.NewConstraintFeedback(pc, fb.DepTypeImported).LogFeedback(i.Logger)
		}

		if version != nil {
			lp := gps.NewLockedProject(pc.Ident, version, nil)
			i.Lock.P = append(i.Lock.P, lp)
			fb.NewLockedProjectFeedback(lp, fb.DepTypeImported).LogFeedback(i.Logger)
		}
	}
}

// isConstraintPinned returns if a constraint is pinned to a specific revision.
func (i *Importer) isConstraintPinned(c gps.Constraint) bool {
	if version, isVersion := c.(gps.Version); isVersion {
		switch version.Type() {
		case gps.IsRevision, gps.IsVersion:
			return true
		}
	}
	return false
}

// testConstraint verifies that the constraint won't invalidate the locked version.
func (i *Importer) testConstraint(c gps.Constraint, v gps.Version) bool {
	// Assume branch constraints are satisfied
	if version, isVersion := c.(gps.Version); isVersion {
		if version.Type() == gps.IsBranch {

			return true
		}
	}

	return c.Matches(v)
}

// convertToConstraint turns a version into a constraint.
// Semver tags are converted to a range with the caret operator.
func (i *Importer) convertToConstraint(v gps.Version) gps.Constraint {
	if v.Type() == gps.IsSemver {
		c, err := gps.NewSemverConstraintIC(v.String())
		if err != nil {
			// This should never fail, because the type is semver.
			// If it does fail somehow, don't let that impact the import.
			return nil
		}
		return c
	}
	return v
}

func (i *Importer) isDefaultSource(projectRoot gps.ProjectRoot, sourceURL string) (bool, error) {
	// this condition is mainly for gopkg.in imports,
	// as some importers specify the repository url as https://gopkg.in/...,
	// but SourceManager.SourceURLsForPath() returns https://github.com/... urls for gopkg.in
	if sourceURL == "https://"+string(projectRoot) {
		return true, nil
	}

	sourceURLs, err := i.SourceManager.SourceURLsForPath(string(projectRoot))
	if err != nil {
		return false, err
	}
	// The first url in the slice will be the default one (usually https://...)
	if len(sourceURLs) > 0 && sourceURL == sourceURLs[0].String() {
		return true, nil
	}

	return false, nil
}
