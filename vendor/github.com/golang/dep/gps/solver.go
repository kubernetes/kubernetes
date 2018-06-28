// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"container/heap"
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/armon/go-radix"
	"github.com/golang/dep/gps/paths"
	"github.com/golang/dep/gps/pkgtree"
	"github.com/pkg/errors"
)

var rootRev = Revision("")

// SolveParameters hold all arguments to a solver run.
//
// Only RootDir and RootPackageTree are absolutely required. A nil Manifest is
// allowed, though it usually makes little sense.
//
// Of these properties, only the Manifest and RootPackageTree are (directly)
// incorporated in memoization hashing.
type SolveParameters struct {
	// The path to the root of the project on which the solver should operate.
	// This should point to the directory that should contain the vendor/
	// directory.
	//
	// In general, it is wise for this to be under an active GOPATH, though it
	// is not (currently) required.
	//
	// A real path to a readable directory is required.
	RootDir string

	// The ProjectAnalyzer is responsible for extracting Manifest and
	// (optionally) Lock information from dependencies. The solver passes it
	// along to its SourceManager's GetManifestAndLock() method as needed.
	//
	// An analyzer is required.
	ProjectAnalyzer ProjectAnalyzer

	// The tree of packages that comprise the root project, as well as the
	// import path that should identify the root of that tree.
	//
	// In most situations, tools should simply pass the result of ListPackages()
	// directly through here.
	//
	// The ImportRoot property must be a non-empty string, and at least one
	// element must be present in the Packages map.
	RootPackageTree pkgtree.PackageTree

	// The root manifest. This contains all the dependency constraints
	// associated with normal Manifests, as well as the particular controls
	// afforded only to the root project.
	//
	// May be nil, but for most cases, that would be unwise.
	Manifest RootManifest

	// The root lock. Optional. Generally, this lock is the output of a previous
	// solve run.
	//
	// If provided, the solver will attempt to preserve the versions specified
	// in the lock, unless ToChange or ChangeAll settings indicate otherwise.
	Lock Lock

	// ToChange is a list of project names that should be changed - that is, any
	// versions specified for those projects in the root lock file should be
	// ignored.
	//
	// Passing ChangeAll has subtly different behavior from enumerating all
	// projects into ToChange. In general, ToChange should *only* be used if the
	// user expressly requested an upgrade for a specific project.
	ToChange []ProjectRoot

	// ChangeAll indicates that all projects should be changed - that is, any
	// versions specified in the root lock file should be ignored.
	ChangeAll bool

	// Downgrade indicates whether the solver will attempt to upgrade (false) or
	// downgrade (true) projects that are not locked, or are marked for change.
	//
	// Upgrading is, by far, the most typical case. The field is named
	// 'Downgrade' so that the bool's zero value corresponds to that most
	// typical case.
	Downgrade bool

	// TraceLogger is the logger to use for generating trace output. If set, the
	// solver will generate informative trace output as it moves through the
	// solving process.
	TraceLogger *log.Logger

	// stdLibFn is the function to use to recognize standard library import paths.
	// Only overridden for tests. Defaults to paths.IsStandardImportPath if nil.
	stdLibFn func(string) bool

	// mkBridgeFn is the function to use to create sourceBridges.
	// Only overridden for tests (so we can run with virtual RootDir).
	// Defaults to mkBridge if nil.
	mkBridgeFn func(*solver, SourceManager, bool) sourceBridge
}

// solver is a CDCL-style constraint solver with satisfiability conditions
// hardcoded to the needs of the Go package management problem space.
type solver struct {
	// The current number of attempts made over the course of this solve. This
	// number increments each time the algorithm completes a backtrack and
	// starts moving forward again.
	attempts int

	// Logger used exclusively for trace output, or nil to suppress.
	tl *log.Logger

	// The function to use to recognize standard library import paths.
	stdLibFn func(string) bool

	// A bridge to the standard SourceManager. The adapter does some local
	// caching of pre-sorted version lists, as well as translation between the
	// full-on ProjectIdentifiers that the solver deals with and the simplified
	// names a SourceManager operates on.
	b sourceBridge

	// A versionUnifier, to facilitate cross-type version comparison and set
	// operations.
	vUnify *versionUnifier

	// A stack containing projects and packages that are currently "selected" -
	// that is, they have passed all satisfiability checks, and are part of the
	// current solution.
	//
	// The *selection type is mostly just a dumb data container; the solver
	// itself is responsible for maintaining that invariant.
	sel *selection

	// The current list of projects that we need to incorporate into the solution in
	// order for the solution to be complete. This list is implemented as a
	// priority queue that places projects least likely to induce errors at the
	// front, in order to minimize the amount of backtracking required to find a
	// solution.
	//
	// Entries are added to and removed from this list by the solver at the same
	// time that the selected queue is updated, either with an addition or
	// removal.
	unsel *unselected

	// A stack of all the currently active versionQueues in the solver. The set
	// of projects represented here corresponds closely to what's in s.sel,
	// although s.sel will always contain the root project, and s.vqs never
	// will. Also, s.vqs is only added to (or popped from during backtracking)
	// when a new project is selected; it is untouched when new packages are
	// added to an existing project.
	vqs []*versionQueue

	// Contains data and constraining information from the root project
	rd rootdata

	// metrics for the current solve run.
	mtr *metrics

	// Indicates whether the solver has been run. It is invalid to run this type
	// of solver more than once.
	hasrun int32
}

func (params SolveParameters) toRootdata() (rootdata, error) {
	if params.ProjectAnalyzer == nil {
		return rootdata{}, badOptsFailure("must provide a ProjectAnalyzer")
	}
	if params.RootDir == "" {
		return rootdata{}, badOptsFailure("params must specify a non-empty root directory")
	}
	if params.RootPackageTree.ImportRoot == "" {
		return rootdata{}, badOptsFailure("params must include a non-empty import root")
	}
	if len(params.RootPackageTree.Packages) == 0 {
		return rootdata{}, badOptsFailure("at least one package must be present in the PackageTree")
	}
	if params.Lock == nil && len(params.ToChange) != 0 {
		return rootdata{}, badOptsFailure(fmt.Sprintf("update specifically requested for %s, but no lock was provided to upgrade from", params.ToChange))
	}

	if params.Manifest == nil {
		params.Manifest = simpleRootManifest{}
	}

	rd := rootdata{
		ir:      params.Manifest.IgnoredPackages(),
		req:     params.Manifest.RequiredPackages(),
		ovr:     params.Manifest.Overrides(),
		rpt:     params.RootPackageTree.Copy(),
		chng:    make(map[ProjectRoot]struct{}),
		rlm:     make(map[ProjectRoot]LockedProject),
		chngall: params.ChangeAll,
		dir:     params.RootDir,
		an:      params.ProjectAnalyzer,
	}

	// Ensure the required and overrides maps are at least initialized
	if rd.req == nil {
		rd.req = make(map[string]bool)
	}
	if rd.ovr == nil {
		rd.ovr = make(ProjectConstraints)
	}

	if rd.ir.Len() > 0 {
		var both []string
		for pkg := range params.Manifest.RequiredPackages() {
			if rd.ir.IsIgnored(pkg) {
				both = append(both, pkg)
			}
		}
		switch len(both) {
		case 0:
			break
		case 1:
			return rootdata{}, badOptsFailure(fmt.Sprintf("%q was given as both a required and ignored package", both[0]))
		default:
			return rootdata{}, badOptsFailure(fmt.Sprintf("multiple packages given as both required and ignored: %s", strings.Join(both, ", ")))
		}
	}

	// Validate no empties in the overrides map
	var eovr []string
	for pr, pp := range rd.ovr {
		if pp.Constraint == nil && pp.Source == "" {
			eovr = append(eovr, string(pr))
		}
	}

	if eovr != nil {
		// Maybe it's a little nitpicky to do this (we COULD proceed; empty
		// overrides have no effect), but this errs on the side of letting the
		// tool/user know there's bad input. Purely as a principle, that seems
		// preferable to silently allowing progress with icky input.
		if len(eovr) > 1 {
			return rootdata{}, badOptsFailure(fmt.Sprintf("Overrides lacked any non-zero properties for multiple project roots: %s", strings.Join(eovr, " ")))
		}
		return rootdata{}, badOptsFailure(fmt.Sprintf("An override was declared for %s, but without any non-zero properties", eovr[0]))
	}

	// Prep safe, normalized versions of root manifest and lock data
	rd.rm = prepManifest(params.Manifest)

	if params.Lock != nil {
		for _, lp := range params.Lock.Projects() {
			rd.rlm[lp.Ident().ProjectRoot] = lp
		}

		// Also keep a prepped one, mostly for the bridge. This is probably
		// wasteful, but only minimally so, and yay symmetry
		rd.rl = prepLock(params.Lock)
	}

	for _, p := range params.ToChange {
		if _, exists := rd.rlm[p]; !exists {
			return rootdata{}, badOptsFailure(fmt.Sprintf("cannot update %s as it is not in the lock", p))
		}
		rd.chng[p] = struct{}{}
	}

	return rd, nil
}

// Prepare readies a Solver for use.
//
// This function reads and validates the provided SolveParameters. If a problem
// with the inputs is detected, an error is returned. Otherwise, a Solver is
// returned, ready to hash and check inputs or perform a solving run.
func Prepare(params SolveParameters, sm SourceManager) (Solver, error) {
	if sm == nil {
		return nil, badOptsFailure("must provide non-nil SourceManager")
	}

	rd, err := params.toRootdata()
	if err != nil {
		return nil, err
	}

	if params.stdLibFn == nil {
		params.stdLibFn = paths.IsStandardImportPath
	}

	s := &solver{
		tl:       params.TraceLogger,
		stdLibFn: params.stdLibFn,
		rd:       rd,
	}

	// Set up the bridge and ensure the root dir is in good, working order
	// before doing anything else.
	if params.mkBridgeFn == nil {
		s.b = mkBridge(s, sm, params.Downgrade)
	} else {
		s.b = params.mkBridgeFn(s, sm, params.Downgrade)
	}
	err = s.b.verifyRootDir(params.RootDir)
	if err != nil {
		return nil, err
	}
	s.vUnify = &versionUnifier{
		b: s.b,
	}

	// Initialize stacks and queues
	s.sel = &selection{
		deps:      make(map[ProjectRoot][]dependency),
		foldRoots: make(map[string]ProjectRoot),
		vu:        s.vUnify,
	}
	s.unsel = &unselected{
		sl:  make([]bimodalIdentifier, 0),
		cmp: s.unselectedComparator,
	}

	return s, nil
}

// A Solver is the main workhorse of gps: given a set of project inputs, it
// performs a constraint solving analysis to develop a complete Solution, or
// else fail with an informative error.
//
// If a Solution is found, an implementing tool may persist it - typically into
// a "lock file" - and/or use it to write out a directory tree of dependencies,
// suitable to be a vendor directory, via CreateVendorTree.
type Solver interface {
	// HashInputs hashes the unique inputs to this solver, returning the hash
	// digest. It is guaranteed that, if the resulting digest is equal to the
	// digest returned from a previous Solution.InputHash(), that that Solution
	// is valid for this Solver's inputs.
	//
	// In such a case, it may not be necessary to run Solve() at all.
	HashInputs() []byte

	// Solve initiates a solving run. It will either abort due to a canceled
	// Context, complete successfully with a Solution, or fail with an
	// informative error.
	//
	// It is generally not allowed that this method be called twice for any
	// given solver.
	Solve(context.Context) (Solution, error)

	// Name returns a string identifying the particular solver backend.
	//
	// Different solvers likely have different invariants, and likely will not
	// have the same result sets for any particular inputs.
	Name() string

	// Version returns an int indicating the version of the solver of the given
	// Name(). Implementations should change their reported version ONLY when
	// the logic is changed in such a way that substantially changes the result
	// set that is possible for a substantial subset of likely inputs.
	//
	// "Substantial" is an imprecise term, and it is used intentionally. There
	// are no easy, general ways of subdividing constraint solving problems such
	// that one can know, a priori, the full impact that subtle algorithmic
	// changes will have on possible result sets. Consequently, we have to fall
	// back on coarser, intuition-based reasoning as to whether a change is
	// large enough that it is likely to be broadly user-visible.
	//
	// This is acceptable, because this value is not used programmatically by
	// the solver in any way. Rather, it is intend for implementing tools to
	// use as a coarse signal to users about compatibility between their tool's
	// version and the current data, typically via persistence to a Lock.
	// Changes to the version number reported should be weighed between
	// confusing teams by having two members' tools continuously rolling back
	// each others' chosen Solutions for no apparent reason, and annoying teams
	// by changing the number for changes so remote that warnings about solver
	// version mismatches become meaningless.
	//
	// Err on the side of caution.
	//
	// Chronology is the only implication of the ordering - that lower version
	// numbers were published before higher numbers.
	Version() int
}

func (s *solver) Name() string {
	return "gps-cdcl"
}

func (s *solver) Version() int {
	return 1
}

// DeductionErrs maps package import path to errors occurring during deduction.
type DeductionErrs map[string]error

func (e DeductionErrs) Error() string {
	return "could not deduce external imports' project roots"
}

// ValidateParams validates the solver parameters to ensure solving can be completed.
func ValidateParams(params SolveParameters, sm SourceManager) error {
	// Ensure that all packages are deducible without issues.
	var deducePkgsGroup sync.WaitGroup
	deductionErrs := make(DeductionErrs)
	var errsMut sync.Mutex

	rd, err := params.toRootdata()
	if err != nil {
		return err
	}

	deducePkg := func(ip string, sm SourceManager) {
		_, err := sm.DeduceProjectRoot(ip)
		if err != nil {
			errsMut.Lock()
			deductionErrs[ip] = err
			errsMut.Unlock()
		}
		deducePkgsGroup.Done()
	}

	for _, ip := range rd.externalImportList(paths.IsStandardImportPath) {
		deducePkgsGroup.Add(1)
		go deducePkg(ip, sm)
	}

	deducePkgsGroup.Wait()

	if len(deductionErrs) > 0 {
		return deductionErrs
	}

	return nil
}

// Solve attempts to find a dependency solution for the given project, as
// represented by the SolveParameters with which this Solver was created.
//
// This is the entry point to the main gps workhorse.
func (s *solver) Solve(ctx context.Context) (Solution, error) {
	// Solving can only be run once per solver.
	if !atomic.CompareAndSwapInt32(&s.hasrun, 0, 1) {
		return nil, errors.New("solve method can only be run once per instance")
	}
	// Make sure the bridge has the context before we start.
	//s.b.ctx = ctx

	// Set up a metrics object
	s.mtr = newMetrics()
	s.vUnify.mtr = s.mtr

	// Prime the queues with the root project
	if err := s.selectRoot(); err != nil {
		return nil, err
	}

	all, err := s.solve(ctx)

	s.mtr.pop()
	var soln solution
	if err == nil {
		soln = solution{
			att:  s.attempts,
			solv: s,
		}
		soln.analyzerInfo = s.rd.an.Info()
		soln.hd = s.HashInputs()

		// Convert ProjectAtoms into LockedProjects
		soln.p = make([]LockedProject, len(all))
		k := 0
		for pa, pl := range all {
			soln.p[k] = pa2lp(pa, pl)
			k++
		}
	}

	s.traceFinish(soln, err)
	if s.tl != nil {
		s.mtr.dump(s.tl)
	}
	return soln, err
}

// solve is the top-level loop for the solving process.
func (s *solver) solve(ctx context.Context) (map[atom]map[string]struct{}, error) {
	// Pull out the donechan once up front so that we're not potentially
	// triggering mutex cycling and channel creation on each iteration.
	donechan := ctx.Done()

	// Main solving loop
	for {
		select {
		case <-donechan:
			return nil, ctx.Err()
		default:
		}

		bmi, has := s.nextUnselected()

		if !has {
			// no more packages to select - we're done.
			break
		}

		// This split is the heart of "bimodal solving": we follow different
		// satisfiability and selection paths depending on whether we've already
		// selected the base project/repo that came off the unselected queue.
		//
		// (If we've already selected the project, other parts of the algorithm
		// guarantee the bmi will contain at least one package from this project
		// that has yet to be selected.)
		if awp, is := s.sel.selected(bmi.id); !is {
			s.mtr.push("new-atom")
			// Analysis path for when we haven't selected the project yet - need
			// to create a version queue.
			queue, err := s.createVersionQueue(bmi)
			if err != nil {
				s.mtr.pop()
				// Err means a failure somewhere down the line; try backtracking.
				s.traceStartBacktrack(bmi, err, false)
				success, berr := s.backtrack(ctx)
				if berr != nil {
					err = berr
				} else if success {
					// backtracking succeeded, move to the next unselected id
					continue
				}
				return nil, err
			}

			if queue.current() == nil {
				panic("canary - queue is empty, but flow indicates success")
			}

			awp := atomWithPackages{
				a: atom{
					id: queue.id,
					v:  queue.current(),
				},
				pl: bmi.pl,
			}
			err = s.selectAtom(awp, false)
			s.mtr.pop()
			if err != nil {
				// Only a released SourceManager should be able to cause this.
				return nil, err
			}

			s.vqs = append(s.vqs, queue)
		} else {
			s.mtr.push("add-atom")
			// We're just trying to add packages to an already-selected project.
			// That means it's not OK to burn through the version queue for that
			// project as we do when first selecting a project, as doing so
			// would upend the guarantees on which all previous selections of
			// the project are based (both the initial one, and any package-only
			// ones).

			// Because we can only safely operate within the scope of the
			// single, currently selected version, we can skip looking for the
			// queue and just use the version given in what came back from
			// s.sel.selected().
			nawp := atomWithPackages{
				a: atom{
					id: bmi.id,
					v:  awp.a.v,
				},
				pl: bmi.pl,
			}

			s.traceCheckPkgs(bmi)
			err := s.check(nawp, true)
			if err != nil {
				s.mtr.pop()
				// Err means a failure somewhere down the line; try backtracking.
				s.traceStartBacktrack(bmi, err, true)
				success, berr := s.backtrack(ctx)
				if berr != nil {
					err = berr
				} else if success {
					// backtracking succeeded, move to the next unselected id
					continue
				}
				return nil, err
			}
			err = s.selectAtom(nawp, true)
			s.mtr.pop()
			if err != nil {
				// Only a released SourceManager should be able to cause this.
				return nil, err
			}

			// We don't add anything to the stack of version queues because the
			// backtracker knows not to pop the vqstack if it backtracks
			// across a pure-package addition.
		}
	}

	// Getting this far means we successfully found a solution. Combine the
	// selected projects and packages.
	projs := make(map[atom]map[string]struct{})

	// Skip the first project. It's always the root, and that shouldn't be
	// included in results.
	for _, sel := range s.sel.projects[1:] {
		pm, exists := projs[sel.a.a]
		if !exists {
			pm = make(map[string]struct{})
			projs[sel.a.a] = pm
		}

		for _, path := range sel.a.pl {
			pm[path] = struct{}{}
		}
	}
	return projs, nil
}

// selectRoot is a specialized selectAtom, used solely to initially
// populate the queues at the beginning of a solve run.
func (s *solver) selectRoot() error {
	s.mtr.push("select-root")
	// Push the root project onto the queue.
	awp := s.rd.rootAtom()
	s.sel.pushSelection(awp, false)

	// If we're looking for root's deps, get it from opts and local root
	// analysis, rather than having the sm do it.
	deps, err := s.intersectConstraintsWithImports(s.rd.combineConstraints(), s.rd.externalImportList(s.stdLibFn))
	if err != nil {
		if contextCanceledOrSMReleased(err) {
			return err
		}
		// TODO(sdboyer) this could well happen; handle it with a more graceful error
		panic(fmt.Sprintf("canary - shouldn't be possible %s", err))
	}

	for _, dep := range deps {
		// If we have no lock, or if this dep isn't in the lock, then prefetch
		// it. See longer explanation in selectAtom() for how we benefit from
		// parallelism here.
		if s.rd.needVersionsFor(dep.Ident.ProjectRoot) {
			go s.b.SyncSourceFor(dep.Ident)
		}

		s.sel.pushDep(dependency{depender: awp.a, dep: dep})
		// Add all to unselected queue
		heap.Push(s.unsel, bimodalIdentifier{id: dep.Ident, pl: dep.pl, fromRoot: true})
	}

	s.traceSelectRoot(s.rd.rpt, deps)
	s.mtr.pop()
	return nil
}

func (s *solver) getImportsAndConstraintsOf(a atomWithPackages) ([]string, []completeDep, error) {
	var err error

	if s.rd.isRoot(a.a.id.ProjectRoot) {
		panic("Should never need to recheck imports/constraints from root during solve")
	}

	// Work through the source manager to get project info and static analysis
	// information.
	m, _, err := s.b.GetManifestAndLock(a.a.id, a.a.v, s.rd.an)
	if err != nil {
		return nil, nil, err
	}

	ptree, err := s.b.ListPackages(a.a.id, a.a.v)
	if err != nil {
		return nil, nil, err
	}

	rm, em := ptree.ToReachMap(true, false, true, s.rd.ir)
	// Use maps to dedupe the unique internal and external packages.
	exmap, inmap := make(map[string]struct{}), make(map[string]struct{})

	for _, pkg := range a.pl {
		inmap[pkg] = struct{}{}
		for _, ipkg := range rm[pkg].Internal {
			inmap[ipkg] = struct{}{}
		}
	}

	var pl []string
	// If lens are the same, then the map must have the same contents as the
	// slice; no need to build a new one.
	if len(inmap) == len(a.pl) {
		pl = a.pl
	} else {
		pl = make([]string, 0, len(inmap))
		for pkg := range inmap {
			pl = append(pl, pkg)
		}
		sort.Strings(pl)
	}

	// Add to the list those packages that are reached by the packages
	// explicitly listed in the atom
	for _, pkg := range a.pl {
		// Skip ignored packages
		if s.rd.ir.IsIgnored(pkg) {
			continue
		}

		ie, exists := rm[pkg]
		if !exists {
			// Missing package here *should* only happen if the target pkg was
			// poisoned; check the errors map.
			if importErr, eexists := em[pkg]; eexists {
				return nil, nil, importErr
			}

			// Nope, it's actually full-on not there.
			return nil, nil, fmt.Errorf("package %s does not exist within project %s", pkg, a.a.id)
		}

		for _, ex := range ie.External {
			exmap[ex] = struct{}{}
		}
	}

	reach := make([]string, 0, len(exmap))
	for pkg := range exmap {
		reach = append(reach, pkg)
	}
	sort.Strings(reach)

	deps := s.rd.ovr.overrideAll(m.DependencyConstraints())
	cd, err := s.intersectConstraintsWithImports(deps, reach)
	return pl, cd, err
}

// intersectConstraintsWithImports takes a list of constraints and a list of
// externally reached packages, and creates a []completeDep that is guaranteed
// to include all packages named by import reach, using constraints where they
// are available, or Any() where they are not.
func (s *solver) intersectConstraintsWithImports(deps []workingConstraint, reach []string) ([]completeDep, error) {
	// Create a radix tree with all the projects we know from the manifest
	xt := radix.New()
	for _, dep := range deps {
		xt.Insert(string(dep.Ident.ProjectRoot), dep)
	}

	// Step through the reached packages; if they have prefix matches in
	// the trie, assume (mostly) it's a correct correspondence.
	dmap := make(map[ProjectRoot]completeDep)
	for _, rp := range reach {
		// If it's a stdlib-shaped package, skip it.
		if s.stdLibFn(rp) {
			continue
		}

		// Look for a prefix match; it'll be the root project/repo containing
		// the reached package
		if pre, idep, match := xt.LongestPrefix(rp); match && isPathPrefixOrEqual(pre, rp) {
			// Match is valid; put it in the dmap, either creating a new
			// completeDep or appending it to the existing one for this base
			// project/prefix.
			dep := idep.(workingConstraint)
			if cdep, exists := dmap[dep.Ident.ProjectRoot]; exists {
				cdep.pl = append(cdep.pl, rp)
				dmap[dep.Ident.ProjectRoot] = cdep
			} else {
				dmap[dep.Ident.ProjectRoot] = completeDep{
					workingConstraint: dep,
					pl:                []string{rp},
				}
			}
			continue
		}

		// No match. Let the SourceManager try to figure out the root
		root, err := s.b.DeduceProjectRoot(rp)
		if err != nil {
			// Nothing we can do if we can't suss out a root
			return nil, err
		}

		// Make a new completeDep with an open constraint, respecting overrides
		pd := s.rd.ovr.override(root, ProjectProperties{Constraint: Any()})

		// Insert the pd into the trie so that further deps from this
		// project get caught by the prefix search
		xt.Insert(string(root), pd)
		// And also put the complete dep into the dmap
		dmap[root] = completeDep{
			workingConstraint: pd,
			pl:                []string{rp},
		}
	}

	// Dump all the deps from the map into the expected return slice
	cdeps := make([]completeDep, 0, len(dmap))
	for _, cdep := range dmap {
		cdeps = append(cdeps, cdep)
	}

	return cdeps, nil
}

func (s *solver) createVersionQueue(bmi bimodalIdentifier) (*versionQueue, error) {
	id := bmi.id
	// If on the root package, there's no queue to make
	if s.rd.isRoot(id.ProjectRoot) {
		return newVersionQueue(id, nil, nil, s.b)
	}

	exists, err := s.b.SourceExists(id)
	if err != nil {
		return nil, err
	}
	if !exists {
		exists, err = s.b.vendorCodeExists(id)
		if err != nil {
			return nil, err
		}
		if exists {
			// Project exists only in vendor
			// FIXME(sdboyer) this just totally doesn't work at all right now
		} else {
			return nil, fmt.Errorf("project '%s' could not be located", id)
		}
	}

	var lockv Version
	if len(s.rd.rlm) > 0 {
		lockv, err = s.getLockVersionIfValid(id)
		if err != nil {
			// Can only get an error here if an upgrade was expressly requested on
			// code that exists only in vendor
			return nil, err
		}
	}

	var prefv Version
	if bmi.fromRoot {
		// If this bmi came from the root, then we want to search through things
		// with a dependency on it in order to see if any have a lock that might
		// express a prefv
		//
		// TODO(sdboyer) nested loop; prime candidate for a cache somewhere
		for _, dep := range s.sel.getDependenciesOn(bmi.id) {
			// Skip the root, of course
			if s.rd.isRoot(dep.depender.id.ProjectRoot) {
				continue
			}

			_, l, err := s.b.GetManifestAndLock(dep.depender.id, dep.depender.v, s.rd.an)
			if err != nil || l == nil {
				// err being non-nil really shouldn't be possible, but the lock
				// being nil is quite likely
				continue
			}

			for _, lp := range l.Projects() {
				if lp.Ident().eq(bmi.id) {
					prefv = lp.Version()
				}
			}
		}

		// OTHER APPROACH - WRONG, BUT MAYBE USEFUL FOR REFERENCE?
		// If this bmi came from the root, then we want to search the unselected
		// queue to see if anything *else* wants this ident, in which case we
		// pick up that prefv
		//for _, bmi2 := range s.unsel.sl {
		//// Take the first thing from the queue that's for the same ident,
		//// and has a non-nil prefv
		//if bmi.id.eq(bmi2.id) {
		//if bmi2.prefv != nil {
		//prefv = bmi2.prefv
		//}
		//}
		//}

	} else {
		// Otherwise, just use the preferred version expressed in the bmi
		prefv = bmi.prefv
	}

	q, err := newVersionQueue(id, lockv, prefv, s.b)
	if err != nil {
		// TODO(sdboyer) this particular err case needs to be improved to be ONLY for cases
		// where there's absolutely nothing findable about a given project name
		return nil, err
	}

	// Hack in support for revisions.
	//
	// By design, revs aren't returned from ListVersion(). Thus, if the dep in
	// the bmi was has a rev constraint, it is (almost) guaranteed to fail, even
	// if that rev does exist in the repo. So, detect a rev and push it into the
	// vq here, instead.
	//
	// Happily, the solver maintains the invariant that constraints on a given
	// ident cannot be incompatible, so we know that if we find one rev, then
	// any other deps will have to also be on that rev (or Any).
	//
	// TODO(sdboyer) while this does work, it bypasses the interface-implied guarantees
	// of the version queue, and is therefore not a great strategy for API
	// coherency. Folding this in to a formal interface would be better.
	if tc, ok := s.sel.getConstraint(bmi.id).(Revision); ok && q.pi[0] != tc {
		// We know this is the only thing that could possibly match, so put it
		// in at the front - if it isn't there already.
		// TODO(sdboyer) existence of the revision is guaranteed by checkRevisionExists(); restore that call.
		q.pi = append([]Version{tc}, q.pi...)
	}

	// Having assembled the queue, search it for a valid version.
	s.traceCheckQueue(q, bmi, false, 1)
	return q, s.findValidVersion(q, bmi.pl)
}

// findValidVersion walks through a versionQueue until it finds a version that
// satisfies the constraints held in the current state of the solver.
//
// The satisfiability checks triggered from here are constrained to operate only
// on those dependencies induced by the list of packages given in the second
// parameter.
func (s *solver) findValidVersion(q *versionQueue, pl []string) error {
	if nil == q.current() {
		// this case should not be reachable, but reflects improper solver state
		// if it is, so panic immediately
		panic("version queue is empty, should not happen")
	}

	faillen := len(q.fails)

	for {
		cur := q.current()
		s.traceInfo("try %s@%s", q.id, cur)
		err := s.check(atomWithPackages{
			a: atom{
				id: q.id,
				v:  cur,
			},
			pl: pl,
		}, false)
		if err == nil {
			// we have a good version, can return safely
			return nil
		}

		if q.advance(err) != nil {
			// Error on advance, have to bail out
			break
		}
		if q.isExhausted() {
			// Queue is empty, bail with error
			break
		}
	}

	s.fail(s.sel.getDependenciesOn(q.id)[0].depender.id)

	// Return a compound error of all the new errors encountered during this
	// attempt to find a new, valid version
	return &noVersionError{
		pn:    q.id,
		fails: q.fails[faillen:],
	}
}

// getLockVersionIfValid finds an atom for the given ProjectIdentifier from the
// root lock, assuming:
//
// 1. A root lock was provided
// 2. The general flag to change all projects was not passed
// 3. A flag to change this particular ProjectIdentifier was not passed
//
// If any of these three conditions are true (or if the id cannot be found in
// the root lock), then no atom will be returned.
func (s *solver) getLockVersionIfValid(id ProjectIdentifier) (Version, error) {
	// If the project is specifically marked for changes, then don't look for a
	// locked version.
	if _, explicit := s.rd.chng[id.ProjectRoot]; explicit || s.rd.chngall {
		// For projects with an upstream or cache repository, it's safe to
		// ignore what's in the lock, because there's presumably more versions
		// to be found and attempted in the repository. If it's only in vendor,
		// though, then we have to try to use what's in the lock, because that's
		// the only version we'll be able to get.
		if exist, _ := s.b.SourceExists(id); exist {
			// Upgrades mean breaking the lock
			s.b.breakLock()
			return nil, nil
		}

		// However, if a change was *expressly* requested for something that
		// exists only in vendor, then that guarantees we don't have enough
		// information to complete a solution. In that case, error out.
		if explicit {
			return nil, &missingSourceFailure{
				goal: id,
				prob: "Cannot upgrade %s, as no source repository could be found.",
			}
		}
	}

	lp, exists := s.rd.rlm[id.ProjectRoot]
	if !exists {
		return nil, nil
	}

	constraint := s.sel.getConstraint(id)
	v := lp.Version()
	if !constraint.Matches(v) {
		var found bool
		if tv, ok := v.(Revision); ok {
			// If we only have a revision from the root's lock, allow matching
			// against other versions that have that revision
			for _, pv := range s.vUnify.pairRevision(id, tv) {
				if constraint.Matches(pv) {
					v = pv
					found = true
					break
				}
			}
			//} else if _, ok := constraint.(Revision); ok {
			//// If the current constraint is itself a revision, and the lock gave
			//// an unpaired version, see if they match up
			////
			//if u, ok := v.(UnpairedVersion); ok {
			//pv := s.sm.pairVersion(id, u)
			//if constraint.Matches(pv) {
			//v = pv
			//found = true
			//}
			//}
		}

		if !found {
			// No match found, which means we're going to be breaking the lock
			// Still return the invalid version so that is included in the trace
			s.b.breakLock()
		}
	}

	return v, nil
}

// backtrack works backwards from the current failed solution to find the next
// solution to try.
func (s *solver) backtrack(ctx context.Context) (bool, error) {
	if len(s.vqs) == 0 {
		// nothing to backtrack to
		return false, nil
	}

	donechan := ctx.Done()
	s.mtr.push("backtrack")
	defer s.mtr.pop()
	for {
		for {
			select {
			case <-donechan:
				return false, ctx.Err()
			default:
			}

			if len(s.vqs) == 0 {
				// no more versions, nowhere further to backtrack
				return false, nil
			}
			if s.vqs[len(s.vqs)-1].failed {
				break
			}

			s.vqs, s.vqs[len(s.vqs)-1] = s.vqs[:len(s.vqs)-1], nil

			// Pop selections off until we get to a project.
			var proj bool
			var awp atomWithPackages
			for !proj {
				var err error
				awp, proj, err = s.unselectLast()
				if err != nil {
					if !contextCanceledOrSMReleased(err) {
						panic(fmt.Sprintf("canary - should only have been able to get a context cancellation or SM release, got %T %s", err, err))
					}
					return false, err
				}
				s.traceBacktrack(awp.bmi(), !proj)
			}
		}

		// Grab the last versionQueue off the list of queues
		q := s.vqs[len(s.vqs)-1]

		// Walk back to the next project. This may entail walking through some
		// package-only selections.
		var proj bool
		var awp atomWithPackages
		for !proj {
			var err error
			awp, proj, err = s.unselectLast()
			if err != nil {
				if !contextCanceledOrSMReleased(err) {
					panic(fmt.Sprintf("canary - should only have been able to get a context cancellation or SM release, got %T %s", err, err))
				}
				return false, err
			}
			s.traceBacktrack(awp.bmi(), !proj)
		}

		if !q.id.eq(awp.a.id) {
			panic("canary - version queue stack and selected project stack are misaligned")
		}

		// Advance the queue past the current version, which we know is bad
		// TODO(sdboyer) is it feasible to make available the failure reason here?
		if q.advance(nil) == nil && !q.isExhausted() {
			// Search for another acceptable version of this failed dep in its queue
			s.traceCheckQueue(q, awp.bmi(), true, 0)
			if s.findValidVersion(q, awp.pl) == nil {
				// Found one! Put it back on the selected queue and stop
				// backtracking

				// reusing the old awp is fine
				awp.a.v = q.current()
				err := s.selectAtom(awp, false)
				if err != nil {
					if !contextCanceledOrSMReleased(err) {
						panic(fmt.Sprintf("canary - should only have been able to get a context cancellation or SM release, got %T %s", err, err))
					}
					return false, err
				}
				break
			}
		}

		s.traceBacktrack(awp.bmi(), false)

		// No solution found; continue backtracking after popping the queue
		// we just inspected off the list
		// GC-friendly pop pointer elem in slice
		s.vqs, s.vqs[len(s.vqs)-1] = s.vqs[:len(s.vqs)-1], nil
	}

	// Backtracking was successful if loop ended before running out of versions
	if len(s.vqs) == 0 {
		return false, nil
	}
	s.attempts++
	return true, nil
}

func (s *solver) nextUnselected() (bimodalIdentifier, bool) {
	if len(s.unsel.sl) > 0 {
		return s.unsel.sl[0], true
	}

	return bimodalIdentifier{}, false
}

func (s *solver) unselectedComparator(i, j int) bool {
	ibmi, jbmi := s.unsel.sl[i], s.unsel.sl[j]
	iname, jname := ibmi.id, jbmi.id

	// Most important thing is pushing package additions ahead of project
	// additions. Package additions can't walk their version queue, so all they
	// do is narrow the possibility of success; better to find out early and
	// fast if they're going to fail than wait until after we've done real work
	// on a project and have to backtrack across it.

	// FIXME the impl here is currently O(n) in the number of selections; it
	// absolutely cannot stay in a hot sorting path like this
	// FIXME while other solver invariants probably protect us from it, this
	// call-out means that it's possible for external state change to invalidate
	// heap invariants.
	_, isel := s.sel.selected(iname)
	_, jsel := s.sel.selected(jname)

	if isel && !jsel {
		return true
	}
	if !isel && jsel {
		return false
	}

	if iname.eq(jname) {
		return false
	}

	_, ilock := s.rd.rlm[iname.ProjectRoot]
	_, jlock := s.rd.rlm[jname.ProjectRoot]

	switch {
	case ilock && !jlock:
		return true
	case !ilock && jlock:
		return false
	case ilock && jlock:
		return iname.Less(jname)
	}

	// Now, sort by number of available versions. This will trigger network
	// activity, but at this point we know that the project we're looking at
	// isn't locked by the root. And, because being locked by root is the only
	// way avoid that call when making a version queue, we know we're gonna have
	// to pay that cost anyway.

	// We can safely ignore an err from listVersions here because, if there is
	// an actual problem, it'll be noted and handled somewhere else saner in the
	// solving algorithm.
	ivl, _ := s.b.listVersions(iname)
	jvl, _ := s.b.listVersions(jname)
	iv, jv := len(ivl), len(jvl)

	// Packages with fewer versions to pick from are less likely to benefit from
	// backtracking, so deal with them earlier in order to minimize the amount
	// of superfluous backtracking through them we do.
	switch {
	case iv == 0 && jv != 0:
		return true
	case iv != 0 && jv == 0:
		return false
	case iv != jv:
		return iv < jv
	}

	// Finally, if all else fails, fall back to comparing by name
	return iname.Less(jname)
}

func (s *solver) fail(id ProjectIdentifier) {
	// TODO(sdboyer) does this need updating, now that we have non-project package
	// selection?

	// skip if the root project
	if !s.rd.isRoot(id.ProjectRoot) {
		// just look for the first (oldest) one; the backtracker will necessarily
		// traverse through and pop off any earlier ones
		for _, vq := range s.vqs {
			if vq.id.eq(id) {
				vq.failed = true
				return
			}
		}
	}
}

// selectAtom pulls an atom into the selection stack, alongside some of
// its contained packages. New resultant dependency requirements are added to
// the unselected priority queue.
//
// Behavior is slightly diffferent if pkgonly is true.
func (s *solver) selectAtom(a atomWithPackages, pkgonly bool) error {
	s.mtr.push("select-atom")
	s.unsel.remove(bimodalIdentifier{
		id: a.a.id,
		pl: a.pl,
	})

	pl, deps, err := s.getImportsAndConstraintsOf(a)
	if err != nil {
		if contextCanceledOrSMReleased(err) {
			return err
		}
		// This shouldn't be possible; other checks should have ensured all
		// packages and deps are present for any argument passed to this method.
		panic(fmt.Sprintf("canary - shouldn't be possible %s", err))
	}
	// Assign the new internal package list into the atom, then push it onto the
	// selection stack
	a.pl = pl
	s.sel.pushSelection(a, pkgonly)

	// If this atom has a lock, pull it out so that we can potentially inject
	// preferred versions into any bmis we enqueue
	//
	// TODO(sdboyer) making this call here could be the first thing to trigger
	// network activity...maybe? if so, can we mitigate by deferring the work to
	// queue consumption time?
	_, l, _ := s.b.GetManifestAndLock(a.a.id, a.a.v, s.rd.an)
	var lmap map[ProjectIdentifier]Version
	if l != nil {
		lmap = make(map[ProjectIdentifier]Version)
		for _, lp := range l.Projects() {
			lmap[lp.Ident()] = lp.Version()
		}
	}

	for _, dep := range deps {
		// Root can come back up here if there's a project-level cycle.
		// Satisfiability checks have already ensured invariants are maintained,
		// so we know we can just skip it here.
		if s.rd.isRoot(dep.Ident.ProjectRoot) {
			continue
		}
		// If this is dep isn't in the lock, do some prefetching. (If it is, we
		// might be able to get away with zero network activity for it, so don't
		// prefetch). This provides an opportunity for some parallelism wins, on
		// two fronts:
		//
		// 1. Because this loop may have multiple deps in it, we could end up
		// simultaneously fetching both in the background while solving proceeds
		//
		// 2. Even if only one dep gets prefetched here, the worst case is that
		// that same dep comes out of the unselected queue next, and we gain a
		// few microseconds before blocking later. Best case, the dep doesn't
		// come up next, but some other dep comes up that wasn't prefetched, and
		// both fetches proceed in parallel.
		if s.rd.needVersionsFor(dep.Ident.ProjectRoot) {
			go s.b.SyncSourceFor(dep.Ident)
		}

		s.sel.pushDep(dependency{depender: a.a, dep: dep})
		// Go through all the packages introduced on this dep, selecting only
		// the ones where the only depper on them is what the preceding line just
		// pushed in. Then, put those into the unselected queue.
		rpm := s.sel.getRequiredPackagesIn(dep.Ident)
		var newp []string
		for _, pkg := range dep.pl {
			// Just one means that the dep we're visiting is the sole importer.
			if rpm[pkg] == 1 {
				newp = append(newp, pkg)
			}
		}

		if len(newp) > 0 {
			// If there was a previously-established alternate source for this
			// dependency, but the current atom did not express one (and getting
			// here means the atom passed the source hot-swapping check - see
			// checkIdentMatches()), then we have to create the new bmi with the
			// alternate source. Otherwise, we end up with two discrete project
			// entries for the project root in the final output, one with the
			// alternate source, and one without. See #969.
			id, _ := s.sel.getIdentFor(dep.Ident.ProjectRoot)
			bmi := bimodalIdentifier{
				id: id,
				pl: newp,
				// This puts in a preferred version if one's in the map, else
				// drops in the zero value (nil)
				prefv: lmap[dep.Ident],
			}
			heap.Push(s.unsel, bmi)
		}
	}

	s.traceSelect(a, pkgonly)
	s.mtr.pop()

	return nil
}

func (s *solver) unselectLast() (atomWithPackages, bool, error) {
	s.mtr.push("unselect")
	defer s.mtr.pop()
	awp, first := s.sel.popSelection()
	heap.Push(s.unsel, bimodalIdentifier{id: awp.a.id, pl: awp.pl})

	_, deps, err := s.getImportsAndConstraintsOf(awp)
	if err != nil {
		if contextCanceledOrSMReleased(err) {
			return atomWithPackages{}, false, err
		}
		// This shouldn't be possible; other checks should have ensured all
		// packages and deps are present for any argument passed to this method.
		panic(fmt.Sprintf("canary - shouldn't be possible %s", err))
	}

	for _, dep := range deps {
		// Skip popping if the dep is the root project, which can occur if
		// there's a project-level import cycle. (This occurs frequently with
		// e.g. kubernetes and docker)
		if s.rd.isRoot(dep.Ident.ProjectRoot) {
			continue
		}
		s.sel.popDep(dep.Ident)

		// if no parents/importers, remove from unselected queue
		if s.sel.depperCount(dep.Ident) == 0 {
			s.unsel.remove(bimodalIdentifier{id: dep.Ident, pl: dep.pl})
		}
	}

	return awp, first, nil
}

// simple (temporary?) helper just to convert atoms into locked projects
func pa2lp(pa atom, pkgs map[string]struct{}) LockedProject {
	lp := LockedProject{
		pi: pa.id,
	}

	switch v := pa.v.(type) {
	case UnpairedVersion:
		lp.v = v
	case Revision:
		lp.r = v
	case versionPair:
		lp.v = v.v
		lp.r = v.r
	default:
		panic("unreachable")
	}

	lp.pkgs = make([]string, len(pkgs))
	k := 0

	pr := string(pa.id.ProjectRoot)
	trim := pr + "/"
	for pkg := range pkgs {
		if pkg == string(pa.id.ProjectRoot) {
			lp.pkgs[k] = "."
		} else {
			lp.pkgs[k] = strings.TrimPrefix(pkg, trim)
		}
		k++
	}
	sort.Strings(lp.pkgs)

	return lp
}

func contextCanceledOrSMReleased(err error) bool {
	return err == context.Canceled || err == context.DeadlineExceeded || err == ErrSourceManagerIsReleased
}
