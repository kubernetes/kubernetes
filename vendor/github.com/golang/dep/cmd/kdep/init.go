// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"flag"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/golang/dep"
	"github.com/golang/dep/gps"
	"github.com/golang/dep/internal/fs"
	"github.com/golang/dep/internal/kdep"
	"github.com/pkg/errors"
)

const initShortHelp = `Set up a new Go project, or migrate an existing one`
const initLongHelp = `
Initialize the project at filepath root by parsing its dependencies, writing
manifest and lock files, and vendoring the dependencies. If root isn't
specified, use the current directory.

When configuration for another dependency management tool is detected, it is
imported into the initial manifest and lock. Use the -skip-tools flag to
disable this behavior. The following external tools are supported:
glide, godep, vndr, govend, gb, gvt, govendor, glock.

Any dependencies that are not constrained by external configuration use the
GOPATH analysis below.

By default, the dependencies are resolved over the network. A version will be
selected from the versions available from the upstream source per the following
algorithm:

 - Tags conforming to semver (sorted by semver rules)
 - Default branch(es) (sorted lexicographically)
 - Non-semver tags (sorted lexicographically)

An alternate mode can be activated by passing -gopath. In this mode, the version
of each dependency will reflect the current state of the GOPATH. If a dependency
doesn't exist in the GOPATH, a version will be selected based on the above
network version selection algorithm.

A Gopkg.toml file will be written with inferred version constraints for all
direct dependencies. Gopkg.lock will be written with precise versions, and
vendor/ will be populated with the precise versions written to Gopkg.lock.
`

func (cmd *initCommand) Name() string      { return "init" }
func (cmd *initCommand) Args() string      { return "[root]" }
func (cmd *initCommand) ShortHelp() string { return initShortHelp }
func (cmd *initCommand) LongHelp() string  { return initLongHelp }
func (cmd *initCommand) Hidden() bool      { return false }

func (cmd *initCommand) Register(fs *flag.FlagSet) {
	fs.BoolVar(&cmd.noExamples, "no-examples", false, "don't include example in Gopkg.toml")
	fs.BoolVar(&cmd.skipTools, "skip-tools", false, "skip importing configuration from other dependency managers")
	fs.BoolVar(&cmd.gopath, "gopath", false, "search in GOPATH for dependencies")
}

type initCommand struct {
	noExamples bool
	skipTools  bool
	gopath     bool
}

func (cmd *initCommand) Run(ctx *kdep.Ctx, args []string) error {
	if len(args) > 1 {
		return errors.Errorf("too many args (%d)", len(args))
	}

	var root string
	if len(args) <= 0 {
		root = ctx.WorkingDir
	} else {
		root = args[0]
		if !filepath.IsAbs(args[0]) {
			root = filepath.Join(ctx.WorkingDir, args[0])
		}
		if err := os.MkdirAll(root, os.FileMode(0777)); err != nil {
			return errors.Wrapf(err, "init failed: unable to create a directory at %s", root)
		}
	}

	p, err := cmd.establishProjectAt(root, ctx)
	if err != nil {
		return err
	}

	sm, err := ctx.SourceManager()
	if err != nil {
		return errors.Wrap(err, "init failed: unable to create a source manager")
	}
	sm.UseDefaultSignalHandling()
	defer sm.Release()

	if ctx.Verbose {
		ctx.Out.Println("Getting direct dependencies...")
	}

	ptree, directDeps, err := p.GetDirectDependencyNames(sm)
	if err != nil {
		return errors.Wrap(err, "init failed: unable to determine direct dependencies")
	}
	if ctx.Verbose {
		ctx.Out.Printf("Checked %d directories for packages.\nFound %d direct dependencies.\n", len(ptree.Packages), len(directDeps))
	}

	// Initialize with imported data, then fill in the gaps using the GOPATH
	rootAnalyzer := newRootAnalyzer(cmd.skipTools, ctx, directDeps, sm)
	p.Manifest, p.Lock, err = rootAnalyzer.InitializeRootManifestAndLock(root, p.ImportRoot)
	if err != nil {
		return errors.Wrap(err, "init failed: unable to prepare an initial manifest and lock for the solver")
	}

	// Set default prune options for go-tests and unused-packages
	p.Manifest.PruneOptions.DefaultOptions = gps.PruneNestedVendorDirs | gps.PruneGoTestFiles | gps.PruneUnusedPackages

	if cmd.gopath {
		gs := newGopathScanner(ctx, directDeps, sm)
		err = gs.InitializeRootManifestAndLock(p.Manifest, p.Lock)
		if err != nil {
			return errors.Wrap(err, "init failed: unable to scan the GOPATH for dependencies")
		}
	}

	rootAnalyzer.skipTools = true // Don't import external config during solve for now
	copyLock := *p.Lock           // Copy lock before solving. Use this to separate new lock projects from solved lock

	params := gps.SolveParameters{
		RootDir:         root,
		RootPackageTree: ptree,
		Manifest:        p.Manifest,
		Lock:            p.Lock,
		ProjectAnalyzer: rootAnalyzer,
	}

	if ctx.Verbose {
		params.TraceLogger = ctx.Err
	}

	if err := ctx.ValidateParams(sm, params); err != nil {
		return errors.Wrapf(err, "init failed: validation of solve parameters failed")
	}

	s, err := gps.Prepare(params, sm)
	if err != nil {
		return errors.Wrap(err, "init failed: unable to prepare the solver")
	}

	soln, err := s.Solve(context.TODO())
	if err != nil {
		err = handleAllTheFailuresOfTheWorld(err)
		return errors.Wrap(err, "init failed: unable to solve the dependency graph")
	}
	p.Lock = dep.LockFromSolution(soln)

	rootAnalyzer.FinalizeRootManifestAndLock(p.Manifest, p.Lock, copyLock)

	// Run gps.Prepare with appropriate constraint solutions from solve run
	// to generate the final lock memo.
	s, err = gps.Prepare(params, sm)
	if err != nil {
		return errors.Wrap(err, "init failed: unable to recalculate the lock digest")
	}

	p.Lock.SolveMeta.InputsDigest = s.HashInputs()

	// Pass timestamp (yyyyMMddHHmmss format) as suffix to backup name.
	vendorbak, err := dep.BackupVendor(filepath.Join(root, "vendor"), time.Now().Format("20060102150405"))
	if err != nil {
		return errors.Wrap(err, "init failed: first backup vendor/, delete it, and then retry the previous command: failed to backup existing vendor directory")
	}
	if vendorbak != "" {
		ctx.Err.Printf("Old vendor backed up to %v", vendorbak)
	}

	sw, err := dep.NewSafeWriter(p.Manifest, nil, p.Lock, dep.VendorAlways, p.Manifest.PruneOptions)
	if err != nil {
		return errors.Wrap(err, "init failed: unable to create a SafeWriter")
	}

	logger := ctx.Err
	if !ctx.Verbose {
		logger = log.New(ioutil.Discard, "", 0)
	}
	if err := sw.Write(root, sm, !cmd.noExamples, logger); err != nil {
		return errors.Wrap(err, "init failed: unable to write the manifest, lock and vendor directory to disk")
	}

	return nil
}

// establishProjectAt attempts to set up the provided path as the root for the
// project to be created.
//
// It checks for being within a GOPATH, that there is no pre-existing manifest
// and lock, and that we can successfully infer the root import path from
// GOPATH.
//
// If successful, it returns a dep.Project, ready for further use.
func (cmd *initCommand) establishProjectAt(root string, ctx *kdep.Ctx) (*dep.Project, error) {
	var err error
	p := new(dep.Project)
	if err = p.SetRoot(root); err != nil {
		return nil, errors.Wrapf(err, "init failed: unable to set the root project to %s", root)
	}

	ctx.GOPATH, err = ctx.DetectProjectGOPATH(p)
	if err != nil {
		return nil, errors.Wrapf(err, "init failed: unable to detect the containing GOPATH")
	}

	mf := filepath.Join(root, dep.ManifestName)
	lf := filepath.Join(root, dep.LockName)

	mok, err := fs.IsRegular(mf)
	if err != nil {
		return nil, errors.Wrapf(err, "init failed: unable to check for an existing manifest at %s", mf)
	}
	if mok {
		return nil, errors.Errorf("init aborted: manifest already exists at %s", mf)
	}

	lok, err := fs.IsRegular(lf)
	if err != nil {
		return nil, errors.Wrapf(err, "init failed: unable to check for an existing lock at %s", lf)
	}
	if lok {
		return nil, errors.Errorf("invalid aborted: lock already exists at %s", lf)
	}

	ip, err := ctx.ImportForAbs(root)
	if err != nil {
		return nil, errors.Wrapf(err, "init failed: unable to determine the import path for the root project %s", root)
	}
	p.ImportRoot = gps.ProjectRoot(ip)

	return p, nil
}
