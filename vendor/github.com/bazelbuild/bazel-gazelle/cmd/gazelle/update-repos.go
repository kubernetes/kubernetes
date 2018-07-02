/* Copyright 2017 The Bazel Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"

	"github.com/bazelbuild/bazel-gazelle/internal/merger"
	"github.com/bazelbuild/bazel-gazelle/internal/repos"
	"github.com/bazelbuild/bazel-gazelle/internal/wspace"
	bf "github.com/bazelbuild/buildtools/build"
)

type updateReposFn func(c *updateReposConfiguration, oldFile *bf.File) error

type updateReposConfiguration struct {
	fn           updateReposFn
	repoRoot     string
	lockFilename string
	importPaths  []string
}

func updateRepos(args []string) error {
	c, err := newUpdateReposConfiguration(args)
	if err != nil {
		return err
	}

	workspacePath := filepath.Join(c.repoRoot, "WORKSPACE")
	content, err := ioutil.ReadFile(workspacePath)
	if err != nil {
		return fmt.Errorf("error reading %q: %v", workspacePath, err)
	}
	f, err := bf.Parse(workspacePath, content)
	if err != nil {
		return fmt.Errorf("error parsing %q: %v", workspacePath, err)
	}
	merger.FixWorkspace(f)

	if err := c.fn(c, f); err != nil {
		return err
	}
	merger.FixLoads(f)
	if err := merger.CheckGazelleLoaded(f); err != nil {
		return err
	}
	if err := ioutil.WriteFile(f.Path, bf.Format(f), 0666); err != nil {
		return fmt.Errorf("error writing %q: %v", f.Path, err)
	}
	return nil
}

func newUpdateReposConfiguration(args []string) (*updateReposConfiguration, error) {
	c := new(updateReposConfiguration)
	fs := flag.NewFlagSet("gazelle", flag.ContinueOnError)
	// Flag will call this on any parse error. Don't print usage unless
	// -h or -help were passed explicitly.
	fs.Usage = func() {}

	fromFileFlag := fs.String("from_file", "", "Gazelle will translate repositories listed in this file into repository rules in WORKSPACE. Currently only dep's Gopkg.lock is supported.")
	repoRootFlag := fs.String("repo_root", "", "path to the root directory of the repository. If unspecified, this is assumed to be the directory containing WORKSPACE.")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			updateReposUsage(fs)
			os.Exit(0)
		}
		// flag already prints the error; don't print it again.
		return nil, errors.New("Try -help for more information")
	}

	// Handle general flags that apply to all subcommands.
	c.repoRoot = *repoRootFlag
	if c.repoRoot == "" {
		if repoRoot, err := wspace.Find("."); err != nil {
			return nil, err
		} else {
			c.repoRoot = repoRoot
		}
	}

	// Handle flags specific to each subcommand.
	switch {
	case *fromFileFlag != "":
		if len(fs.Args()) != 0 {
			return nil, fmt.Errorf("Got %d positional arguments with -from_file; wanted 0.\nTry -help for more information.", len(fs.Args()))
		}
		c.fn = importFromLockFile
		c.lockFilename = *fromFileFlag

	default:
		if len(fs.Args()) == 0 {
			return nil, fmt.Errorf("No repositories specified\nTry -help for more information.")
		}
		c.fn = updateImportPaths
		c.importPaths = fs.Args()
	}

	return c, nil
}

func updateReposUsage(fs *flag.FlagSet) {
	fmt.Fprint(os.Stderr, `usage:

# Add/update repositories by import path
gazelle update-repos example.com/repo1 example.com/repo2

# Import repositories from lock file
gazelle update-repos -from_file=file

The update-repos command updates repository rules in the WORKSPACE file.
update-repos can add or update repositories explicitly by import path.
update-repos can also import repository rules from a vendoring tool's lock
file (currently only deps' Gopkg.lock is supported).

FLAGS:

`)
}

func updateImportPaths(c *updateReposConfiguration, f *bf.File) error {
	rs := repos.ListRepositories(f)
	rc := repos.NewRemoteCache(rs)

	genRules := make([]bf.Expr, len(c.importPaths))
	errs := make([]error, len(c.importPaths))
	var wg sync.WaitGroup
	wg.Add(len(c.importPaths))
	for i, imp := range c.importPaths {
		go func(i int) {
			defer wg.Done()
			repo, err := repos.UpdateRepo(rc, imp)
			if err != nil {
				errs[i] = err
				return
			}
			repo.Remote = "" // don't set these explicitly
			repo.VCS = ""
			rule := repos.GenerateRule(repo)
			genRules[i] = rule
		}(i)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	merger.MergeFile(genRules, nil, f, merger.RepoAttrs)
	return nil
}

func importFromLockFile(c *updateReposConfiguration, f *bf.File) error {
	genRules, err := repos.ImportRepoRules(c.lockFilename)
	if err != nil {
		return err
	}

	merger.MergeFile(genRules, nil, f, merger.RepoAttrs)
	return nil
}
