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
	"os"
	"path/filepath"
	"sync"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/merger"
	"github.com/bazelbuild/bazel-gazelle/internal/repos"
	"github.com/bazelbuild/bazel-gazelle/internal/rule"
)

type updateReposFn func(c *updateReposConfig, oldFile *rule.File, kinds map[string]rule.KindInfo) error

type updateReposConfig struct {
	fn           updateReposFn
	lockFilename string
	importPaths  []string
}

const updateReposName = "_update-repos"

func getUpdateReposConfig(c *config.Config) *updateReposConfig {
	return c.Exts[updateReposName].(*updateReposConfig)
}

type updateReposConfigurer struct{}

func (_ *updateReposConfigurer) RegisterFlags(fs *flag.FlagSet, cmd string, c *config.Config) {
	uc := &updateReposConfig{}
	c.Exts[updateReposName] = uc
	fs.StringVar(&uc.lockFilename, "from_file", "", "Gazelle will translate repositories listed in this file into repository rules in WORKSPACE. Currently only dep's Gopkg.lock is supported.")
}

func (_ *updateReposConfigurer) CheckFlags(fs *flag.FlagSet, c *config.Config) error {
	uc := getUpdateReposConfig(c)
	switch {
	case uc.lockFilename != "":
		if len(fs.Args()) != 0 {
			return fmt.Errorf("Got %d positional arguments with -from_file; wanted 0.\nTry -help for more information.", len(fs.Args()))
		}
		uc.fn = importFromLockFile

	default:
		if len(fs.Args()) == 0 {
			return fmt.Errorf("No repositories specified\nTry -help for more information.")
		}
		uc.fn = updateImportPaths
		uc.importPaths = fs.Args()
	}
	return nil
}

func (_ *updateReposConfigurer) KnownDirectives() []string { return nil }

func (_ *updateReposConfigurer) Configure(c *config.Config, rel string, f *rule.File) {}

func updateRepos(args []string) error {
	cexts := make([]config.Configurer, 0, len(languages)+2)
	cexts = append(cexts, &config.CommonConfigurer{}, &updateReposConfigurer{})
	kinds := make(map[string]rule.KindInfo)
	loads := []rule.LoadInfo{}
	for _, lang := range languages {
		cexts = append(cexts, lang)
		loads = append(loads, lang.Loads()...)
		for kind, info := range lang.Kinds() {
			kinds[kind] = info
		}
	}
	c, err := newUpdateReposConfiguration(args, cexts)
	if err != nil {
		return err
	}
	uc := getUpdateReposConfig(c)

	workspacePath := filepath.Join(c.RepoRoot, "WORKSPACE")
	f, err := rule.LoadFile(workspacePath, "")
	if err != nil {
		return fmt.Errorf("error loading %q: %v", workspacePath, err)
	}
	merger.FixWorkspace(f)

	if err := uc.fn(uc, f, kinds); err != nil {
		return err
	}
	merger.FixLoads(f, loads)
	if err := merger.CheckGazelleLoaded(f); err != nil {
		return err
	}
	if err := f.Save(f.Path); err != nil {
		return fmt.Errorf("error writing %q: %v", f.Path, err)
	}
	return nil
}

func newUpdateReposConfiguration(args []string, cexts []config.Configurer) (*config.Config, error) {
	c := config.New()
	fs := flag.NewFlagSet("gazelle", flag.ContinueOnError)
	// Flag will call this on any parse error. Don't print usage unless
	// -h or -help were passed explicitly.
	fs.Usage = func() {}
	for _, cext := range cexts {
		cext.RegisterFlags(fs, "update-repos", c)
	}
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			updateReposUsage(fs)
			return nil, err
		}
		// flag already prints the error; don't print it again.
		return nil, errors.New("Try -help for more information")
	}
	for _, cext := range cexts {
		if err := cext.CheckFlags(fs, c); err != nil {
			return nil, err
		}
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

func updateImportPaths(c *updateReposConfig, f *rule.File, kinds map[string]rule.KindInfo) error {
	rs := repos.ListRepositories(f)
	rc := repos.NewRemoteCache(rs)

	genRules := make([]*rule.Rule, len(c.importPaths))
	errs := make([]error, len(c.importPaths))
	var wg sync.WaitGroup
	wg.Add(len(c.importPaths))
	for i, imp := range c.importPaths {
		go func(i int, imp string) {
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
		}(i, imp)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	merger.MergeFile(f, nil, genRules, merger.PreResolve, kinds)
	return nil
}

func importFromLockFile(c *updateReposConfig, f *rule.File, kinds map[string]rule.KindInfo) error {
	genRules, err := repos.ImportRepoRules(c.lockFilename)
	if err != nil {
		return err
	}

	merger.MergeFile(f, nil, genRules, merger.PreResolve, kinds)
	return nil
}
