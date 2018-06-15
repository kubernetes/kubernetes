// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"go/build"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/golang/dep"
	"github.com/golang/dep/gps"
	"github.com/golang/dep/gps/paths"
	"github.com/golang/dep/gps/pkgtree"
	"github.com/golang/dep/internal/kdep"
	"github.com/pkg/errors"
)

const ensureShortHelp = `Ensure a dependency is safely vendored in the project`
const ensureLongHelp = `
Project spec:

  <import path>[:alt source URL][@<constraint>]


Ensure gets a project into a complete, reproducible, and likely compilable state:

  * All non-stdlib imports are fulfilled
  * All rules in Gopkg.toml are respected
  * Gopkg.lock records precise versions for all dependencies
  * vendor/ is populated according to Gopkg.lock

Ensure has fast techniques to determine that some of these steps may be
unnecessary. If that determination is made, ensure may skip some steps. Flags
may be passed to bypass these checks; -vendor-only will allow an out-of-date
Gopkg.lock to populate vendor/, and -no-vendor will update Gopkg.lock (if
needed), but never touch vendor/.

The effect of passing project spec arguments varies slightly depending on the
combination of flags that are passed.


Examples:

  dep ensure                                 Populate vendor from existing Gopkg.toml and Gopkg.lock
  dep ensure -add github.com/pkg/foo         Introduce a named dependency at its newest version
  dep ensure -add github.com/pkg/foo@^1.0.1  Introduce a named dependency with a particular constraint

For more detailed usage examples, see dep ensure -examples.
`
const ensureExamples = `
dep ensure

    Solve the project's dependency graph, and place all dependencies in the
    vendor folder. If a dependency is in the lock file, use the version
    specified there. Otherwise, use the most recent version that can satisfy the
    constraints in the manifest file.

dep ensure -vendor-only

    Write vendor/ from an existing Gopkg.lock file, without first verifying that
    the lock is in sync with imports and Gopkg.toml. (This may be useful for
    e.g. strategically layering a Docker images)

dep ensure -add github.com/pkg/foo github.com/pkg/foo/bar

    Introduce one or more dependencies, at their newest version, ensuring that
    specific packages are present in Gopkg.lock and vendor/. Also, append a
    corresponding constraint to Gopkg.toml.

    Note: packages introduced in this way will disappear on the next "dep
    ensure" if an import statement is not added first.

dep ensure -add github.com/pkg/foo/subpkg@1.0.0 bitbucket.org/pkg/bar/baz@master

    Append version constraints to Gopkg.toml for one or more packages, if no
    such rules already exist.

    If the named packages are not already imported, also ensure they are present
    in Gopkg.lock and vendor/. As in the preceding example, packages introduced
    in this way will disappear on the next "dep ensure" if an import statement
    is not added first.

dep ensure -add github.com/pkg/foo:git.internal.com/alt/foo

    Specify an alternate location to treat as the upstream source for a dependency.

dep ensure -update github.com/pkg/foo github.com/pkg/bar

    Update a list of dependencies to the latest versions allowed by Gopkg.toml,
    ignoring any versions recorded in Gopkg.lock. Write the results to
    Gopkg.lock and vendor/.

dep ensure -update

    Update all dependencies to the latest versions allowed by Gopkg.toml,
    ignoring any versions recorded in Gopkg.lock. Update the lock file with any
    changes. (NOTE: Not recommended. Updating one/some dependencies at a time is
    preferred.)

dep ensure -update -no-vendor

    As above, but only modify Gopkg.lock; leave vendor/ unchanged.

dep ensure -no-vendor -dry-run

    This fails with a non zero exit code if Gopkg.lock is not up to date with
    the Gopkg.toml or the project imports. It can be useful to run this during
    CI to check if Gopkg.lock is up to date.

`

var (
	errUpdateArgsValidation = errors.New("update arguments validation failed")
	errAddDepsFailed        = errors.New("adding dependencies failed")
)

func (cmd *ensureCommand) Name() string { return "ensure" }
func (cmd *ensureCommand) Args() string {
	return "[-update | -add] [-no-vendor | -vendor-only] [-dry-run] [-v] [<spec>...]"
}
func (cmd *ensureCommand) ShortHelp() string { return ensureShortHelp }
func (cmd *ensureCommand) LongHelp() string  { return ensureLongHelp }
func (cmd *ensureCommand) Hidden() bool      { return false }

func (cmd *ensureCommand) Register(fs *flag.FlagSet) {
	fs.BoolVar(&cmd.examples, "examples", false, "print detailed usage examples")
	fs.BoolVar(&cmd.update, "update", false, "update the named dependencies (or all, if none are named) in Gopkg.lock to the latest allowed by Gopkg.toml")
	fs.BoolVar(&cmd.add, "add", false, "add new dependencies, or populate Gopkg.toml with constraints for existing dependencies")
	fs.BoolVar(&cmd.vendorOnly, "vendor-only", false, "populate vendor/ from Gopkg.lock without updating it first")
	fs.BoolVar(&cmd.noVendor, "no-vendor", false, "update Gopkg.lock (if needed), but do not update vendor/")
	fs.BoolVar(&cmd.dryRun, "dry-run", false, "only report the changes that would be made")
}

type ensureCommand struct {
	examples   bool
	update     bool
	add        bool
	noVendor   bool
	vendorOnly bool
	dryRun     bool
}

func (cmd *ensureCommand) Run(ctx *kdep.Ctx, args []string) error {
	if cmd.examples {
		ctx.Err.Println(strings.TrimSpace(ensureExamples))
		return nil
	}

	if err := cmd.validateFlags(); err != nil {
		return err
	}

	p, err := ctx.LoadProject()
	if err != nil {
		return err
	}

	sm, err := ctx.SourceManager()
	if err != nil {
		return err
	}
	sm.UseDefaultSignalHandling()
	defer sm.Release()

	if err := dep.ValidateProjectRoots(ctx.Ctx, p.Manifest.Manifest, sm); err != nil {
		return err
	}

	params := p.MakeParams()
	if ctx.Verbose {
		params.TraceLogger = ctx.Err
	}

	if cmd.vendorOnly {
		return cmd.runVendorOnly(ctx, args, p, sm, params)
	}

	params.RootPackageTree, err = p.ParseRootPackageTree()
	if err != nil {
		return err
	}

	if fatal, err := checkErrors(params.RootPackageTree.Packages, p.Manifest.IgnoredPackages()); err != nil {
		if fatal {
			return err
		} else if ctx.Verbose {
			ctx.Out.Println(err)
		}
	}
	if ineffs := p.FindIneffectualConstraints(sm); len(ineffs) > 0 {
		ctx.Err.Printf("Warning: the following project(s) have [[constraint]] stanzas in %s:\n\n", dep.ManifestName)
		for _, ineff := range ineffs {
			ctx.Err.Println("  ✗ ", ineff)
		}
		// TODO(sdboyer) lazy wording, it does not mention ignores at all
		ctx.Err.Printf("\nHowever, these projects are not direct dependencies of the current project:\n")
		ctx.Err.Printf("they are not imported in any .go files, nor are they in the 'required' list in\n")
		ctx.Err.Printf("%s. Dep only applies [[constraint]] rules to direct dependencies, so\n", dep.ManifestName)
		ctx.Err.Printf("these rules will have no effect.\n\n")
		ctx.Err.Printf("Either import/require packages from these projects so that they become direct\n")
		ctx.Err.Printf("dependencies, or convert each [[constraint]] to an [[override]] to enforce rules\n")
		ctx.Err.Printf("on these projects, if they happen to be transitive dependencies,\n\n")
	}

	if cmd.add {
		return cmd.runAdd(ctx, args, p, sm, params)
	} else if cmd.update {
		return cmd.runUpdate(ctx, args, p, sm, params)
	}
	return cmd.runDefault(ctx, args, p, sm, params)
}

func (cmd *ensureCommand) validateFlags() error {
	if cmd.add && cmd.update {
		return errors.New("cannot pass both -add and -update")
	}

	if cmd.vendorOnly {
		if cmd.update {
			return errors.New("-vendor-only makes -update a no-op; cannot pass them together")
		}
		if cmd.add {
			return errors.New("-vendor-only makes -add a no-op; cannot pass them together")
		}
		if cmd.noVendor {
			// TODO(sdboyer) can't think of anything not snarky right now
			return errors.New("really?")
		}
	}
	return nil
}

func (cmd *ensureCommand) vendorBehavior() dep.VendorBehavior {
	if cmd.noVendor {
		return dep.VendorNever
	}
	return dep.VendorOnChanged
}

func (cmd *ensureCommand) runDefault(ctx *kdep.Ctx, args []string, p *kdep.Project, sm gps.SourceManager, params gps.SolveParameters) error {
	// Bare ensure doesn't take any args.
	if len(args) != 0 {
		return errors.New("dep ensure only takes spec arguments with -add or -update")
	}

	if err := ctx.ValidateParams(sm, params); err != nil {
		return err
	}

	solver, err := gps.Prepare(params, sm)
	if err != nil {
		return errors.Wrap(err, "prepare solver")
	}

	if p.Lock != nil && bytes.Equal(p.Lock.InputsDigest(), solver.HashInputs()) {
		// Memo matches, so there's probably nothing to do.
		if ctx.Verbose {
			ctx.Out.Printf("%s was already in sync with imports and %s\n", dep.LockName, dep.ManifestName)
		}

		if cmd.noVendor {
			// The user said not to touch vendor/, so definitely nothing to do.
			return nil
		}

		// TODO(sdboyer) The desired behavior at this point is to determine
		// whether it's necessary to write out vendor, or if it's already
		// consistent with the lock. However, we haven't yet determined what
		// that "verification" is supposed to look like (#121); in the meantime,
		// we unconditionally write out vendor/ so that `dep ensure`'s behavior
		// is maximally compatible with what it will eventually become.
		sw, err := dep.NewSafeWriter(nil, p.Lock, p.Lock, dep.VendorAlways, p.Manifest.PruneOptions)
		if err != nil {
			return err
		}

		if cmd.dryRun {
			return sw.PrintPreparedActions(ctx.Out, ctx.Verbose)
		}

		logger := ctx.Err
		if !ctx.Verbose {
			logger = log.New(ioutil.Discard, "", 0)
		}

		err = sw.Write(p.AbsRoot, sm, true, logger)
		if err != nil {
			return errors.Wrap(err, "grouped write of manifest, lock and vendor")
		}

		return p.HackExtraVendorEntries()
	}

	if cmd.noVendor && cmd.dryRun {
		return errors.New("Gopkg.lock was not up to date")
	}

	solution, err := solver.Solve(context.TODO())
	if err != nil {
		return handleAllTheFailuresOfTheWorld(err)
	}

	sw, err := dep.NewSafeWriter(nil, p.Lock, dep.LockFromSolution(solution), cmd.vendorBehavior(), p.Manifest.PruneOptions)
	if err != nil {
		return err
	}
	if cmd.dryRun {
		return sw.PrintPreparedActions(ctx.Out, ctx.Verbose)
	}

	logger := ctx.Err
	if !ctx.Verbose {
		logger = log.New(ioutil.Discard, "", 0)
	}

	err = sw.Write(p.AbsRoot, sm, false, logger)
	if err != nil {
		return errors.Wrap(err, "grouped write of manifest, lock and vendor")
	}

	err = p.HackExtraVendorEntries()
	if err != nil {
		return errors.Wrap(err, "failed to generate extra vendor entries")
	}

	err = p.HackGodepsCompat(solution)
	if err != nil {
		return errors.Wrap(err, "failed to generate Godeps.json")
	}

	return nil
}

func (cmd *ensureCommand) runVendorOnly(ctx *kdep.Ctx, args []string, p *kdep.Project, sm gps.SourceManager, params gps.SolveParameters) error {
	if len(args) != 0 {
		return errors.Errorf("dep ensure -vendor-only only populates vendor/ from %s; it takes no spec arguments", dep.LockName)
	}

	if p.Lock == nil {
		return errors.Errorf("no %s exists from which to populate vendor/", dep.LockName)
	}
	// Pass the same lock as old and new so that the writer will observe no
	// difference and choose not to write it out.
	sw, err := dep.NewSafeWriter(nil, p.Lock, p.Lock, dep.VendorAlways, p.Manifest.PruneOptions)
	if err != nil {
		return err
	}

	if cmd.dryRun {
		return sw.PrintPreparedActions(ctx.Out, ctx.Verbose)
	}

	logger := ctx.Err
	if !ctx.Verbose {
		logger = log.New(ioutil.Discard, "", 0)
	}
	return errors.WithMessage(sw.Write(p.AbsRoot, sm, true, logger), "grouped write of manifest, lock and vendor")
}

func (cmd *ensureCommand) runUpdate(ctx *kdep.Ctx, args []string, p *kdep.Project, sm gps.SourceManager, params gps.SolveParameters) error {
	if p.Lock == nil {
		return errors.Errorf("-update works by updating the versions recorded in %s, but %s does not exist", dep.LockName, dep.LockName)
	}

	if err := ctx.ValidateParams(sm, params); err != nil {
		return err
	}

	// We'll need to discard this prepared solver as later work changes params,
	// but solver preparation is cheap and worth doing up front in order to
	// perform the fastpath check of hash comparison.
	solver, err := gps.Prepare(params, sm)
	if err != nil {
		return errors.Wrap(err, "fastpath solver prepare")
	}

	// Compare the hashes. If they're not equal, bail out and ask the user to
	// run a straight `dep ensure` before updating. This is handholding the
	// user a bit, but the extra effort required is minimal, and it ensures the
	// user is isolating variables in the event of solve problems (was it the
	// "pending" changes, or the -update that caused the problem?).
	if !bytes.Equal(p.Lock.InputsDigest(), solver.HashInputs()) {
		ctx.Out.Printf("Warning: %s is out of sync with %s or the project's imports.", dep.LockName, dep.ManifestName)
	}

	// When -update is specified without args, allow every dependency to change
	// versions, regardless of the lock file.
	if len(args) == 0 {
		params.ChangeAll = true
	}

	if err := validateUpdateArgs(ctx, args, p, sm, &params); err != nil {
		return err
	}

	// Re-prepare a solver now that our params are complete.
	solver, err = gps.Prepare(params, sm)
	if err != nil {
		return errors.Wrap(err, "fastpath solver prepare")
	}
	solution, err := solver.Solve(context.TODO())
	if err != nil {
		// TODO(sdboyer) special handling for warning cases as described in spec
		// - e.g., named projects did not upgrade even though newer versions
		// were available.
		return handleAllTheFailuresOfTheWorld(err)
	}

	sw, err := dep.NewSafeWriter(nil, p.Lock, dep.LockFromSolution(solution), cmd.vendorBehavior(), p.Manifest.PruneOptions)
	if err != nil {
		return err
	}
	if cmd.dryRun {
		return sw.PrintPreparedActions(ctx.Out, ctx.Verbose)
	}

	logger := ctx.Err
	if !ctx.Verbose {
		logger = log.New(ioutil.Discard, "", 0)
	}
	err = sw.Write(p.AbsRoot, sm, false, logger)
	if err != nil {
		return errors.Wrap(err, "grouped write of manifest, lock and vendor")
	}

	err = p.HackExtraVendorEntries()
	if err != nil {
		return errors.Wrap(err, "failed to generate extra vendor entries")
	}

	err = p.HackGodepsCompat(solution)
	if err != nil {
		return errors.Wrap(err, "failed to generate Godeps.json")
	}

	return nil
}

func (cmd *ensureCommand) runAdd(ctx *kdep.Ctx, args []string, p *kdep.Project, sm gps.SourceManager, params gps.SolveParameters) error {
	if len(args) == 0 {
		return errors.New("must specify at least one project or package to -add")
	}

	if err := ctx.ValidateParams(sm, params); err != nil {
		return err
	}

	// We'll need to discard this prepared solver as later work changes params,
	// but solver preparation is cheap and worth doing up front in order to
	// perform the fastpath check of hash comparison.
	solver, err := gps.Prepare(params, sm)
	if err != nil {
		return errors.Wrap(err, "fastpath solver prepare")
	}

	// Compare the hashes. If they're not equal, bail out and ask the user to
	// run a straight `dep ensure` before updating. This is handholding the
	// user a bit, but the extra effort required is minimal, and it ensures the
	// user is isolating variables in the event of solve problems (was it the
	// "pending" changes, or the -add that caused the problem?).
	if p.Lock != nil && !bytes.Equal(p.Lock.InputsDigest(), solver.HashInputs()) {
		ctx.Out.Printf("Warning: %s is out of sync with %s or the project's imports.", dep.LockName, dep.ManifestName)
	}

	rm, _ := params.RootPackageTree.ToReachMap(true, true, false, p.Manifest.IgnoredPackages())

	// TODO(sdboyer) re-enable this once we ToReachMap() intelligently filters out normally-excluded (_*, .*), dirs from errmap
	//rm, errmap := params.RootPackageTree.ToReachMap(true, true, false, p.Manifest.IgnoredPackages())
	// Having some problematic internal packages isn't cause for termination,
	// but the user needs to be warned.
	//for fail, err := range errmap {
	//if _, is := err.Err.(*build.NoGoError); !is {
	//ctx.Err.Printf("Warning: %s, %s", fail, err)
	//}
	//}

	// Compile unique sets of 1) all external packages imported or required, and
	// 2) the project roots under which they fall.
	exmap := make(map[string]bool)
	exrmap := make(map[gps.ProjectRoot]bool)

	for _, ex := range append(rm.FlattenFn(paths.IsStandardImportPath), p.Manifest.Required...) {
		exmap[ex] = true
		root, err := sm.DeduceProjectRoot(ex)
		if err != nil {
			// This should be very uncommon to hit, as it entails that we
			// couldn't deduce the root for an import, but that some previous
			// solve run WAS able to deduce the root. It's most likely to occur
			// if the user has e.g. not connected to their organization's VPN,
			// and thus cannot access an internal go-get metadata service.
			return errors.Wrapf(err, "could not deduce project root for %s", ex)
		}
		exrmap[root] = true
	}

	// Note: these flags are only partially used by the latter parts of the
	// algorithm; rather, it relies on inference. However, they remain in their
	// entirety as future needs may make further use of them, being a handy,
	// terse way of expressing the original context of the arg inputs.
	type addType uint8
	const (
		// Straightforward case - this induces a temporary require, and thus
		// a warning message about it being ephemeral.
		isInManifest addType = 1 << iota
		// If solving works, we'll pull this constraint from the in-memory
		// manifest (where we recorded it earlier) and then append it to the
		// manifest on disk.
		isInImportsWithConstraint
		// If solving works, we'll extract a constraint from the lock and
		// append it into the manifest on disk, similar to init's behavior.
		isInImportsNoConstraint
		// This gets a message AND a hoist from the solution up into the
		// manifest on disk.
		isInNeither
	)

	type addInstruction struct {
		id         gps.ProjectIdentifier
		ephReq     map[string]bool
		constraint gps.Constraint
		typ        addType
	}
	addInstructions := make(map[gps.ProjectRoot]addInstruction)

	// A mutex for limited access to addInstructions by goroutines.
	var mutex sync.Mutex

	// Channel for receiving all the errors.
	errCh := make(chan error, len(args))

	var wg sync.WaitGroup

	ctx.Out.Println("Fetching sources...")

	for i, arg := range args {
		wg.Add(1)

		if ctx.Verbose {
			ctx.Err.Printf("(%d/%d) %s\n", i+1, len(args), arg)
		}

		go func(arg string) {
			defer wg.Done()

			pc, path, err := getProjectConstraint(arg, sm)
			if err != nil {
				// TODO(sdboyer) ensure these errors are contextualized in a sensible way for -add
				errCh <- err
				return
			}

			// check if the the parsed path is the current root path
			if strings.EqualFold(string(p.ImportRoot), string(pc.Ident.ProjectRoot)) {
				errCh <- errors.New("cannot add current project to itself")
				return
			}

			inManifest := p.Manifest.HasConstraintsOn(pc.Ident.ProjectRoot)
			inImports := exmap[string(pc.Ident.ProjectRoot)]
			if inManifest && inImports {
				errCh <- errors.Errorf("nothing to -add, %s is already in %s and the project's direct imports or required list", pc.Ident.ProjectRoot, dep.ManifestName)
				return
			}

			err = sm.SyncSourceFor(pc.Ident)
			if err != nil {
				errCh <- errors.Wrapf(err, "failed to fetch source for %s", pc.Ident.ProjectRoot)
				return
			}

			someConstraint := !gps.IsAny(pc.Constraint) || pc.Ident.Source != ""

			// Obtain a lock for addInstructions
			mutex.Lock()
			defer mutex.Unlock()
			instr, has := addInstructions[pc.Ident.ProjectRoot]
			if has {
				// Multiple packages from the same project were specified as
				// arguments; make sure they agree on declared constraints.
				// TODO(sdboyer) until we have a general method for checking constraint equality, only allow one to declare
				if someConstraint {
					if !gps.IsAny(instr.constraint) || instr.id.Source != "" {
						errCh <- errors.Errorf("can only specify rules once per project being added; rules were given at least twice for %s", pc.Ident.ProjectRoot)
						return
					}
					instr.constraint = pc.Constraint
					instr.id = pc.Ident
				}
			} else {
				instr.ephReq = make(map[string]bool)
				instr.constraint = pc.Constraint
				instr.id = pc.Ident
			}

			if inManifest {
				if someConstraint {
					errCh <- errors.Errorf("%s already contains rules for %s, cannot specify a version constraint or alternate source", dep.ManifestName, path)
					return
				}

				instr.ephReq[path] = true
				instr.typ |= isInManifest
			} else if inImports {
				if !someConstraint {
					if exmap[path] {
						errCh <- errors.Errorf("%s is already imported or required, so -add is only valid with a constraint", path)
						return
					}

					// No constraints, but the package isn't imported; require it.
					// TODO(sdboyer) this case seems like it's getting overly specific and risks muddying the water more than it helps
					instr.ephReq[path] = true
					instr.typ |= isInImportsNoConstraint
				} else {
					// Don't require on this branch if the path was a ProjectRoot;
					// most common here will be the user adding constraints to
					// something they already imported, and if they specify the
					// root, there's a good chance they don't actually want to
					// require the project's root package, but are just trying to
					// indicate which project should receive the constraints.
					if !exmap[path] && string(pc.Ident.ProjectRoot) != path {
						instr.ephReq[path] = true
					}
					instr.typ |= isInImportsWithConstraint
				}
			} else {
				instr.typ |= isInNeither
				instr.ephReq[path] = true
			}

			addInstructions[pc.Ident.ProjectRoot] = instr
		}(arg)
	}

	wg.Wait()
	close(errCh)

	// Newline after printing the fetching source output.
	ctx.Err.Println()

	// Log all the errors.
	if len(errCh) > 0 {
		ctx.Err.Printf("Failed to add the dependencies:\n\n")
		for err := range errCh {
			ctx.Err.Println("  ✗", err.Error())
		}
		ctx.Err.Println()
		return errAddDepsFailed
	}

	// We're now sure all of our add instructions are individually and mutually
	// valid, so it's safe to begin modifying the input parameters.
	for pr, instr := range addInstructions {
		// The arg processing logic above only adds to the ephReq list if
		// that package definitely needs to be on that list, so we don't
		// need to check instr.typ here - if it's in instr.ephReq, it
		// definitely needs to be added to the manifest's required list.
		for path := range instr.ephReq {
			p.Manifest.Required = append(p.Manifest.Required, path)
		}

		// Only two branches can possibly be adding rules, though the
		// isInNeither case may or may not have an empty constraint.
		if instr.typ&(isInNeither|isInImportsWithConstraint) != 0 {
			p.Manifest.Constraints[pr] = gps.ProjectProperties{
				Source:     instr.id.Source,
				Constraint: instr.constraint,
			}
		}
	}

	// Re-prepare a solver now that our params are complete.
	solver, err = gps.Prepare(params, sm)
	if err != nil {
		return errors.Wrap(err, "fastpath solver prepare")
	}
	solution, err := solver.Solve(context.TODO())
	if err != nil {
		// TODO(sdboyer) detect if the failure was specifically about some of the -add arguments
		return handleAllTheFailuresOfTheWorld(err)
	}

	// Prep post-actions and feedback from adds.
	var reqlist []string
	appender := dep.NewManifest()

	for pr, instr := range addInstructions {
		for path := range instr.ephReq {
			reqlist = append(reqlist, path)
		}

		if instr.typ&isInManifest == 0 {
			var pp gps.ProjectProperties
			var found bool
			for _, proj := range solution.Projects() {
				// We compare just ProjectRoot instead of the whole
				// ProjectIdentifier here because an empty source on the input side
				// could have been converted into a source by the solver.
				if proj.Ident().ProjectRoot == pr {
					found = true
					pp = getProjectPropertiesFromVersion(proj.Version())
					break
				}
			}
			if !found {
				panic(fmt.Sprintf("unreachable: solution did not contain -add argument %s, but solver did not fail", pr))
			}
			pp.Source = instr.id.Source

			if !gps.IsAny(instr.constraint) {
				pp.Constraint = instr.constraint
			}
			appender.Constraints[pr] = pp
		}
	}

	extra, err := appender.MarshalTOML()
	if err != nil {
		return errors.Wrap(err, "could not marshal manifest into TOML")
	}
	sort.Strings(reqlist)

	sw, err := dep.NewSafeWriter(nil, p.Lock, dep.LockFromSolution(solution), dep.VendorOnChanged, p.Manifest.PruneOptions)
	if err != nil {
		return err
	}

	if cmd.dryRun {
		return sw.PrintPreparedActions(ctx.Out, ctx.Verbose)
	}

	logger := ctx.Err
	if !ctx.Verbose {
		logger = log.New(ioutil.Discard, "", 0)
	}
	if err := errors.Wrap(sw.Write(p.AbsRoot, sm, true, logger), "grouped write of manifest, lock and vendor"); err != nil {
		return err
	}

	// FIXME(sdboyer) manifest writes ABSOLUTELY need verification - follow up!
	f, err := os.OpenFile(filepath.Join(p.AbsRoot, dep.ManifestName), os.O_APPEND|os.O_WRONLY, 0666)
	if err != nil {
		return errors.Wrapf(err, "opening %s failed", dep.ManifestName)
	}

	if _, err := f.Write(extra); err != nil {
		f.Close()
		return errors.Wrapf(err, "writing to %s failed", dep.ManifestName)
	}

	switch len(reqlist) {
	case 0:
		// nothing to tell the user
	case 1:
		if cmd.noVendor {
			ctx.Out.Printf("%q is not imported by your project, and has been temporarily added to %s.\n", reqlist[0], dep.LockName)
			ctx.Out.Printf("If you run \"dep ensure\" again before actually importing it, it will disappear from %s. Running \"dep ensure -vendor-only\" is safe, and will guarantee it is present in vendor/.", dep.LockName)
		} else {
			ctx.Out.Printf("%q is not imported by your project, and has been temporarily added to %s and vendor/.\n", reqlist[0], dep.LockName)
			ctx.Out.Printf("If you run \"dep ensure\" again before actually importing it, it will disappear from %s and vendor/.", dep.LockName)
		}
	default:
		if cmd.noVendor {
			ctx.Out.Printf("The following packages are not imported by your project, and have been temporarily added to %s:\n", dep.LockName)
			ctx.Out.Printf("\t%s\n", strings.Join(reqlist, "\n\t"))
			ctx.Out.Printf("If you run \"dep ensure\" again before actually importing them, they will disappear from %s. Running \"dep ensure -vendor-only\" is safe, and will guarantee they are present in vendor/.", dep.LockName)
		} else {
			ctx.Out.Printf("The following packages are not imported by your project, and have been temporarily added to %s and vendor/:\n", dep.LockName)
			ctx.Out.Printf("\t%s\n", strings.Join(reqlist, "\n\t"))
			ctx.Out.Printf("If you run \"dep ensure\" again before actually importing them, they will disappear from %s and vendor/.", dep.LockName)
		}
	}

	return errors.Wrapf(f.Close(), "closing %s", dep.ManifestName)
}

func getProjectConstraint(arg string, sm gps.SourceManager) (gps.ProjectConstraint, string, error) {
	emptyPC := gps.ProjectConstraint{
		Constraint: gps.Any(), // default to any; avoids panics later
	}

	// try to split on '@'
	// When there is no `@`, use any version
	var versionStr string
	atIndex := strings.Index(arg, "@")
	if atIndex > 0 {
		parts := strings.SplitN(arg, "@", 2)
		arg = parts[0]
		versionStr = parts[1]
	}

	// TODO: if we decide to keep equals.....

	// split on colon if there is a network location
	var source string
	colonIndex := strings.Index(arg, ":")
	if colonIndex > 0 {
		parts := strings.SplitN(arg, ":", 2)
		arg = parts[0]
		source = parts[1]
	}

	pr, err := sm.DeduceProjectRoot(arg)
	if err != nil {
		return emptyPC, "", errors.Wrapf(err, "could not infer project root from dependency path: %s", arg) // this should go through to the user
	}

	pi := gps.ProjectIdentifier{ProjectRoot: pr, Source: source}
	c, err := sm.InferConstraint(versionStr, pi)
	if err != nil {
		return emptyPC, "", err
	}
	return gps.ProjectConstraint{Ident: pi, Constraint: c}, arg, nil
}

func checkErrors(m map[string]pkgtree.PackageOrErr, ignore *pkgtree.IgnoredRuleset) (fatal bool, err error) {
	var (
		noGoErrors    int
		pkgtreeErrors = make(pkgtreeErrs, 0, len(m))
	)

	for ip, poe := range m {
		if ignore.IsIgnored(ip) {
			continue
		}

		if poe.Err != nil {
			switch poe.Err.(type) {
			case *build.NoGoError:
				noGoErrors++
			default:
				pkgtreeErrors = append(pkgtreeErrors, poe.Err)
			}
		}
	}

	// If pkgtree was empty or all dirs lacked any Go code, return an error.
	if len(m) == 0 || len(m) == noGoErrors {
		return true, errors.New("no dirs contained any Go code")
	}

	// If all dirs contained build errors, return an error.
	if len(m) == len(pkgtreeErrors) {
		return true, errors.New("all dirs contained build errors")
	}

	// If all directories either had no Go files or caused a build error, return an error.
	if len(m) == len(pkgtreeErrors)+noGoErrors {
		return true, pkgtreeErrors
	}

	// If m contained some errors, return a warning with those errors.
	if len(pkgtreeErrors) > 0 {
		return false, pkgtreeErrors
	}

	return false, nil
}

type pkgtreeErrs []error

func (e pkgtreeErrs) Error() string {
	errs := make([]string, 0, len(e))

	for _, err := range e {
		errs = append(errs, err.Error())
	}

	return fmt.Sprintf("found %d errors in the package tree:\n%s", len(e), strings.Join(errs, "\n"))
}

func validateUpdateArgs(ctx *kdep.Ctx, args []string, p *kdep.Project, sm gps.SourceManager, params *gps.SolveParameters) error {
	// Channel for receiving all the valid arguments.
	argsCh := make(chan string, len(args))

	// Channel for receiving all the validation errors.
	errCh := make(chan error, len(args))

	var wg sync.WaitGroup

	// Allow any of specified project versions to change, regardless of the lock
	// file.
	for _, arg := range args {
		wg.Add(1)

		go func(arg string) {
			defer wg.Done()

			// Ensure the provided path has a deducible project root.
			pc, path, err := getProjectConstraint(arg, sm)
			if err != nil {
				// TODO(sdboyer) ensure these errors are contextualized in a sensible way for -update
				errCh <- err
				return
			}
			if path != string(pc.Ident.ProjectRoot) {
				// TODO(sdboyer): does this really merit an abortive error?
				errCh <- errors.Errorf("%s is not a project root, try %s instead", path, pc.Ident.ProjectRoot)
				return
			}

			if !p.Lock.HasProjectWithRoot(pc.Ident.ProjectRoot) {
				errCh <- errors.Errorf("%s is not present in %s, cannot -update it", pc.Ident.ProjectRoot, dep.LockName)
				return
			}

			if pc.Ident.Source != "" {
				errCh <- errors.Errorf("cannot specify alternate sources on -update (%s)", pc.Ident.Source)
				return
			}

			if !gps.IsAny(pc.Constraint) {
				var baseConstraint gps.Constraint
				for proj, c := range p.Manifest.Constraints {
					if proj == pc.Ident.ProjectRoot {
						baseConstraint = c.Constraint
					}
				}
				if baseConstraint == nil {
					errCh <- errors.Errorf("version constraint %s passed for %s, but -update requires constraints declared in %s", pc.Constraint, pc.Ident.ProjectRoot, dep.ManifestName)
				}

				if !pc.Constraint.MatchesAny(baseConstraint) {
					errCh <- errors.Errorf("version constraint %s is incompatible with base constraint %s", pc.Constraint, baseConstraint)
				}

				props := p.Manifest.Constraints[pc.Ident.ProjectRoot]
				props.Constraint = baseConstraint.Intersect(pc.Constraint)
				p.Manifest.Constraints[pc.Ident.ProjectRoot] = props
				// tweak the argument: pass only the project root, since we
				// modified the constraint in Manifest.
				argsCh <- string(pc.Ident.ProjectRoot)
				return
			}

			// Valid argument.
			argsCh <- arg
		}(arg)
	}

	wg.Wait()
	close(errCh)
	close(argsCh)

	// Log all the errors.
	if len(errCh) > 0 {
		ctx.Err.Printf("Invalid arguments passed to ensure -update:\n\n")
		for err := range errCh {
			ctx.Err.Println("  ✗", err.Error())
		}
		ctx.Err.Println()
		return errUpdateArgsValidation
	}

	// Add all the valid arguments to solve params.
	for arg := range argsCh {
		params.ToChange = append(params.ToChange, gps.ProjectRoot(arg))
	}

	return nil
}
