// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

// check performs constraint checks on the provided atom. The set of checks
// differ slightly depending on whether the atom is pkgonly, or if it's the
// entire project being added for the first time.
//
// The goal is to determine whether selecting the atom would result in a state
// where all the solver requirements are still satisfied.
func (s *solver) check(a atomWithPackages, pkgonly bool) error {
	pa := a.a
	if nilpa == pa {
		// This shouldn't be able to happen, but if it does, it unequivocally
		// indicates a logical bug somewhere, so blowing up is preferable
		panic("canary - checking version of empty ProjectAtom")
	}

	s.mtr.push("satisfy")
	var err error
	defer func() {
		if err != nil {
			s.traceInfo(err)
		}
		s.mtr.pop()
	}()

	// If we're pkgonly, then base atom was already determined to be allowable,
	// so we can skip the checkAtomAllowable step.
	if !pkgonly {
		if err = s.checkAtomAllowable(pa); err != nil {
			return err
		}
	}

	if err = s.checkRequiredPackagesExist(a); err != nil {
		return err
	}

	var deps []completeDep
	_, deps, err = s.getImportsAndConstraintsOf(a)
	if err != nil {
		// An err here would be from the package fetcher; pass it straight back
		return err
	}

	// TODO(sdboyer) this deps list contains only packages not already selected
	// from the target atom (assuming one is selected at all). It's fine for
	// now, but won't be good enough when we get around to doing static
	// analysis.
	for _, dep := range deps {
		if err = s.checkIdentMatches(a, dep); err != nil {
			return err
		}
		if err = s.checkRootCaseConflicts(a, dep); err != nil {
			return err
		}
		if err = s.checkDepsConstraintsAllowable(a, dep); err != nil {
			return err
		}
		if err = s.checkDepsDisallowsSelected(a, dep); err != nil {
			return err
		}
		if err = s.checkRevisionExists(a, dep); err != nil {
			return err
		}
		if err = s.checkPackageImportsFromDepExist(a, dep); err != nil {
			return err
		}

		// TODO(sdboyer) add check that fails if adding this atom would create a loop
	}

	return nil
}

// checkAtomAllowable ensures that an atom itself is acceptable with respect to
// the constraints established by the current solution.
func (s *solver) checkAtomAllowable(pa atom) error {
	constraint := s.sel.getConstraint(pa.id)
	if s.vUnify.matches(pa.id, constraint, pa.v) {
		return nil
	}
	// TODO(sdboyer) collect constraint failure reason (wait...aren't we, below?)

	deps := s.sel.getDependenciesOn(pa.id)
	var failparent []dependency
	for _, dep := range deps {
		if !s.vUnify.matches(pa.id, dep.dep.Constraint, pa.v) {
			s.fail(dep.depender.id)
			failparent = append(failparent, dep)
		}
	}

	err := &versionNotAllowedFailure{
		goal:       pa,
		failparent: failparent,
		c:          constraint,
	}

	return err
}

// checkRequiredPackagesExist ensures that all required packages enumerated by
// existing dependencies on this atom are actually present in the atom.
func (s *solver) checkRequiredPackagesExist(a atomWithPackages) error {
	ptree, err := s.b.ListPackages(a.a.id, a.a.v)
	if err != nil {
		// TODO(sdboyer) handle this more gracefully
		return err
	}

	deps := s.sel.getDependenciesOn(a.a.id)
	fp := make(map[string]errDeppers)
	// We inspect these in a bit of a roundabout way, in order to incrementally
	// build up the failure we'd return if there is, indeed, a missing package.
	// TODO(sdboyer) rechecking all of these every time is wasteful. Is there a shortcut?
	for _, dep := range deps {
		for _, pkg := range dep.dep.pl {
			if errdep, seen := fp[pkg]; seen {
				errdep.deppers = append(errdep.deppers, dep.depender)
				fp[pkg] = errdep
			} else {
				perr, has := ptree.Packages[pkg]
				if !has || perr.Err != nil {
					fp[pkg] = errDeppers{
						err:     perr.Err,
						deppers: []atom{dep.depender},
					}
				}
			}
		}
	}

	if len(fp) > 0 {
		return &checkeeHasProblemPackagesFailure{
			goal:    a.a,
			failpkg: fp,
		}
	}
	return nil
}

// checkDepsConstraintsAllowable checks that the constraints of an atom on a
// given dep are valid with respect to existing constraints.
func (s *solver) checkDepsConstraintsAllowable(a atomWithPackages, cdep completeDep) error {
	dep := cdep.workingConstraint
	constraint := s.sel.getConstraint(dep.Ident)
	// Ensure the constraint expressed by the dep has at least some possible
	// intersection with the intersection of existing constraints.
	if s.vUnify.matchesAny(dep.Ident, constraint, dep.Constraint) {
		return nil
	}

	siblings := s.sel.getDependenciesOn(dep.Ident)
	// No admissible versions - visit all siblings and identify the disagreement(s)
	var failsib []dependency
	var nofailsib []dependency
	for _, sibling := range siblings {
		if !s.vUnify.matchesAny(dep.Ident, sibling.dep.Constraint, dep.Constraint) {
			s.fail(sibling.depender.id)
			failsib = append(failsib, sibling)
		} else {
			nofailsib = append(nofailsib, sibling)
		}
	}

	return &disjointConstraintFailure{
		goal:      dependency{depender: a.a, dep: cdep},
		failsib:   failsib,
		nofailsib: nofailsib,
		c:         constraint,
	}
}

// checkDepsDisallowsSelected ensures that an atom's constraints on a particular
// dep are not incompatible with the version of that dep that's already been
// selected.
func (s *solver) checkDepsDisallowsSelected(a atomWithPackages, cdep completeDep) error {
	dep := cdep.workingConstraint
	selected, exists := s.sel.selected(dep.Ident)
	if exists && !s.vUnify.matches(dep.Ident, dep.Constraint, selected.a.v) {
		s.fail(dep.Ident)

		return &constraintNotAllowedFailure{
			goal: dependency{depender: a.a, dep: cdep},
			v:    selected.a.v,
		}
	}
	return nil
}

// checkIdentMatches ensures that the LocalName of a dep introduced by an atom,
// has the same Source as what's already been selected (assuming anything's been
// selected).
//
// In other words, this ensures that the solver never simultaneously selects two
// identifiers with the same local name, but that disagree about where their
// network source is.
func (s *solver) checkIdentMatches(a atomWithPackages, cdep completeDep) error {
	dep := cdep.workingConstraint
	if curid, has := s.sel.getIdentFor(dep.Ident.ProjectRoot); has && !curid.equiv(dep.Ident) {
		deps := s.sel.getDependenciesOn(a.a.id)
		// Fail all the other deps, as there's no way atom can ever be
		// compatible with them
		for _, d := range deps {
			s.fail(d.depender.id)
		}

		return &sourceMismatchFailure{
			shared:   dep.Ident.ProjectRoot,
			sel:      deps,
			current:  curid.normalizedSource(),
			mismatch: dep.Ident.normalizedSource(),
			prob:     a.a,
		}
	}

	return nil
}

// checkRootCaseConflicts ensures that the ProjectRoot specified in the completeDep
// does not have case conflicts with any existing dependencies.
//
// We only need to check the ProjectRoot, rather than any packages therein, as
// the later check for package existence is case-sensitive.
func (s *solver) checkRootCaseConflicts(a atomWithPackages, cdep completeDep) error {
	pr := cdep.workingConstraint.Ident.ProjectRoot
	hasConflict, current := s.sel.findCaseConflicts(pr)
	if !hasConflict {
		return nil
	}

	curid, _ := s.sel.getIdentFor(current)
	deps := s.sel.getDependenciesOn(curid)
	for _, d := range deps {
		s.fail(d.depender.id)
	}

	// If a project has multiple packages that import each other, we treat that
	// as establishing a canonical case variant for the ProjectRoot. It's possible,
	// however, that that canonical variant is not the same one that others
	// imported it under. If that's the situation, then we'll have arrived here
	// when visiting the project, not its dependers, having misclassified its
	// internal imports as external. That means the atomWithPackages will
	// be the wrong case variant induced by the importers, and the cdep will be
	// a link pointing back at the canonical case variant.
	//
	// If this is the case, use a special failure, wrongCaseFailure, that
	// makes a stronger statement as to the correctness of case variants.
	//
	// TODO(sdboyer) This approach to marking failure is less than great, as
	// this will mark the current atom as failed, as well, causing the
	// backtracker to work through it. While that could prove fruitful, it's
	// quite likely just to be wasted effort. Addressing this - if that's a good
	// idea - would entail creating another path back out of checking to enable
	// backjumping directly to the incorrect importers.
	if current == a.a.id.ProjectRoot {
		return &wrongCaseFailure{
			correct: pr,
			goal:    dependency{depender: a.a, dep: cdep},
			badcase: deps,
		}
	}

	return &caseMismatchFailure{
		goal:    dependency{depender: a.a, dep: cdep},
		current: current,
		failsib: deps,
	}
}

// checkPackageImportsFromDepExist ensures that, if the dep is already selected,
// the newly-required set of packages being placed on it exist and are valid.
func (s *solver) checkPackageImportsFromDepExist(a atomWithPackages, cdep completeDep) error {
	sel, is := s.sel.selected(cdep.workingConstraint.Ident)
	if !is {
		// dep is not already selected; nothing to do
		return nil
	}

	ptree, err := s.b.ListPackages(sel.a.id, sel.a.v)
	if err != nil {
		// TODO(sdboyer) handle this more gracefully
		return err
	}

	e := &depHasProblemPackagesFailure{
		goal: dependency{
			depender: a.a,
			dep:      cdep,
		},
		v:    sel.a.v,
		prob: make(map[string]error),
	}

	for _, pkg := range cdep.pl {
		perr, has := ptree.Packages[pkg]
		if !has || perr.Err != nil {
			if has {
				e.prob[pkg] = perr.Err
			} else {
				e.prob[pkg] = nil
			}
		}
	}

	if len(e.prob) > 0 {
		return e
	}
	return nil
}

// checkRevisionExists ensures that if a dependency is constrained by a
// revision, that that revision actually exists.
func (s *solver) checkRevisionExists(a atomWithPackages, cdep completeDep) error {
	r, isrev := cdep.Constraint.(Revision)
	if !isrev {
		// Constraint is not a revision; nothing to do
		return nil
	}

	present, _ := s.b.RevisionPresentIn(cdep.Ident, r)
	if present {
		return nil
	}

	return &nonexistentRevisionFailure{
		goal: dependency{
			depender: a.a,
			dep:      cdep,
		},
		r: r,
	}
}
