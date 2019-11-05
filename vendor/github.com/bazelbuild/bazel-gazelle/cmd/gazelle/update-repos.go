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
	"sort"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/config"
	"github.com/bazelbuild/bazel-gazelle/language"
	"github.com/bazelbuild/bazel-gazelle/merger"
	"github.com/bazelbuild/bazel-gazelle/repo"
	"github.com/bazelbuild/bazel-gazelle/rule"
)

type updateReposConfig struct {
	repoFilePath  string
	importPaths   []string
	macroFileName string
	macroDefName  string
	pruneRules    bool
	workspace     *rule.File
	repoFileMap   map[string]*rule.File
}

const updateReposName = "_update-repos"

func getUpdateReposConfig(c *config.Config) *updateReposConfig {
	return c.Exts[updateReposName].(*updateReposConfig)
}

type updateReposConfigurer struct{}

type macroFlag struct {
	macroFileName *string
	macroDefName  *string
}

func (f macroFlag) Set(value string) error {
	args := strings.Split(value, "%")
	if len(args) != 2 {
		return fmt.Errorf("Failure parsing to_macro: %s, expected format is macroFile%%defName", value)
	}
	if strings.HasPrefix(args[0], "..") {
		return fmt.Errorf("Failure parsing to_macro: %s, macro file path %s should not start with \"..\"", value, args[0])
	}
	*f.macroFileName = args[0]
	*f.macroDefName = args[1]
	return nil
}

func (f macroFlag) String() string {
	return ""
}

func (*updateReposConfigurer) RegisterFlags(fs *flag.FlagSet, cmd string, c *config.Config) {
	uc := &updateReposConfig{}
	c.Exts[updateReposName] = uc
	fs.StringVar(&uc.repoFilePath, "from_file", "", "Gazelle will translate repositories listed in this file into repository rules in WORKSPACE or a .bzl macro function. Gopkg.lock and go.mod files are supported")
	fs.Var(macroFlag{macroFileName: &uc.macroFileName, macroDefName: &uc.macroDefName}, "to_macro", "Tells Gazelle to write repository rules into a .bzl macro function rather than the WORKSPACE file. . The expected format is: macroFile%defName")
	fs.BoolVar(&uc.pruneRules, "prune", false, "When enabled, Gazelle will remove rules that no longer have equivalent repos in the Gopkg.lock/go.mod file. Can only used with -from_file.")
}

func (*updateReposConfigurer) CheckFlags(fs *flag.FlagSet, c *config.Config) error {
	uc := getUpdateReposConfig(c)
	switch {
	case uc.repoFilePath != "":
		if len(fs.Args()) != 0 {
			return fmt.Errorf("got %d positional arguments with -from_file; wanted 0.\nTry -help for more information.", len(fs.Args()))
		}

	default:
		if len(fs.Args()) == 0 {
			return fmt.Errorf("no repositories specified\nTry -help for more information.")
		}
		if uc.pruneRules {
			return fmt.Errorf("the -prune option can only be used with -from_file")
		}
		uc.importPaths = fs.Args()
	}

	var err error
	workspacePath := filepath.Join(c.RepoRoot, "WORKSPACE")
	uc.workspace, err = rule.LoadWorkspaceFile(workspacePath, "")
	if err != nil {
		return fmt.Errorf("loading WORKSPACE file: %v", err)
	}
	c.Repos, uc.repoFileMap, err = repo.ListRepositories(uc.workspace)
	if err != nil {
		return fmt.Errorf("loading WORKSPACE file: %v", err)
	}

	return nil
}

func (*updateReposConfigurer) KnownDirectives() []string { return nil }

func (*updateReposConfigurer) Configure(c *config.Config, rel string, f *rule.File) {}

func updateRepos(args []string) (err error) {
	// Build configuration with all languages.
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

	// TODO(jayconrod): move Go-specific RemoteCache logic to language/go.
	var knownRepos []repo.Repo
	for _, r := range c.Repos {
		if r.Kind() == "go_repository" {
			knownRepos = append(knownRepos, repo.Repo{
				Name:     r.Name(),
				GoPrefix: r.AttrString("importpath"),
				Remote:   r.AttrString("remote"),
				VCS:      r.AttrString("vcs"),
			})
		}
	}
	rc, cleanup := repo.NewRemoteCache(knownRepos)
	defer func() {
		if cerr := cleanup(); err == nil && cerr != nil {
			err = cerr
		}
	}()

	// Fix the workspace file with each language.
	for _, lang := range languages {
		lang.Fix(c, uc.workspace)
	}

	// Generate rules from command language arguments or by importing a file.
	var gen, empty []*rule.Rule
	if uc.repoFilePath == "" {
		gen, err = updateRepoImports(c, rc)
	} else {
		gen, empty, err = importRepos(c, rc)
	}
	if err != nil {
		return err
	}

	// Organize generated and empty rules by file. A rule should go into the file
	// it came from (by name). New rules should go into WORKSPACE or the file
	// specified with -to_macro.
	var newGen []*rule.Rule
	genForFiles := make(map[*rule.File][]*rule.Rule)
	emptyForFiles := make(map[*rule.File][]*rule.Rule)
	for _, r := range gen {
		f := uc.repoFileMap[r.Name()]
		if f != nil {
			genForFiles[f] = append(genForFiles[f], r)
		} else {
			newGen = append(newGen, r)
		}
	}
	for _, r := range empty {
		f := uc.repoFileMap[r.Name()]
		if f == nil {
			panic(fmt.Sprintf("empty rule %q for deletion that was not found", r.Name()))
		}
		emptyForFiles[f] = append(emptyForFiles[f], r)
	}

	var newGenFile *rule.File
	var macroPath string
	if uc.macroFileName != "" {
		macroPath = filepath.Join(c.RepoRoot, filepath.Clean(uc.macroFileName))
	}
	for f := range genForFiles {
		if macroPath == "" && filepath.Base(f.Path) == "WORKSPACE" ||
			macroPath != "" && f.Path == macroPath && f.DefName == uc.macroDefName {
			newGenFile = f
			break
		}
	}
	if newGenFile == nil {
		if uc.macroFileName == "" {
			newGenFile = uc.workspace
		} else {
			var err error
			newGenFile, err = rule.LoadMacroFile(macroPath, "", uc.macroDefName)
			if os.IsNotExist(err) {
				newGenFile, err = rule.EmptyMacroFile(macroPath, "", uc.macroDefName)
				if err != nil {
					return fmt.Errorf("error creating %q: %v", macroPath, err)
				}
			} else if err != nil {
				return fmt.Errorf("error loading %q: %v", macroPath, err)
			}
		}
	}
	genForFiles[newGenFile] = append(genForFiles[newGenFile], newGen...)

	// Merge rules and fix loads in each file.
	seenFile := make(map[*rule.File]bool)
	sortedFiles := make([]*rule.File, 0, len(genForFiles))
	for f := range genForFiles {
		if !seenFile[f] {
			seenFile[f] = true
			sortedFiles = append(sortedFiles, f)
		}
	}
	for f := range emptyForFiles {
		if !seenFile[f] {
			seenFile[f] = true
			sortedFiles = append(sortedFiles, f)
		}
	}
	sort.Slice(sortedFiles, func(i, j int) bool {
		if cmp := strings.Compare(sortedFiles[i].Path, sortedFiles[j].Path); cmp != 0 {
			return cmp < 0
		}
		return sortedFiles[i].DefName < sortedFiles[j].DefName
	})

	updatedFiles := make(map[string]*rule.File)
	for _, f := range sortedFiles {
		merger.MergeFile(f, emptyForFiles[f], genForFiles[f], merger.PreResolve, kinds)
		merger.FixLoads(f, loads)
		if f == uc.workspace {
			if err := merger.CheckGazelleLoaded(f); err != nil {
				return err
			}
		}
		f.Sync()
		if uf, ok := updatedFiles[f.Path]; ok {
			uf.SyncMacroFile(f)
		} else {
			updatedFiles[f.Path] = f
		}
	}
	for _, f := range sortedFiles {
		if uf := updatedFiles[f.Path]; uf != nil {
			if err := uf.Save(uf.Path); err != nil {
				return err
			}
			delete(updatedFiles, f.Path)
		}
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
	fs.PrintDefaults()
}

func updateRepoImports(c *config.Config, rc *repo.RemoteCache) (gen []*rule.Rule, err error) {
	// TODO(jayconrod): let the user pick the language with a command line flag.
	// For now, only use the first language that implements the interface.
	uc := getUpdateReposConfig(c)
	var updater language.RepoUpdater
	for _, lang := range languages {
		if u, ok := lang.(language.RepoUpdater); ok {
			updater = u
			break
		}
	}
	if updater == nil {
		return nil, fmt.Errorf("no languages can update repositories")
	}
	res := updater.UpdateRepos(language.UpdateReposArgs{
		Config:  c,
		Imports: uc.importPaths,
		Cache:   rc,
	})
	return res.Gen, res.Error
}

func importRepos(c *config.Config, rc *repo.RemoteCache) (gen, empty []*rule.Rule, err error) {
	uc := getUpdateReposConfig(c)
	importSupported := false
	var importer language.RepoImporter
	for _, lang := range languages {
		if i, ok := lang.(language.RepoImporter); ok {
			importSupported = true
			if i.CanImport(uc.repoFilePath) {
				importer = i
				break
			}
		}
	}
	if importer == nil {
		if importSupported {
			return nil, nil, fmt.Errorf("unknown file format: %s", uc.repoFilePath)
		} else {
			return nil, nil, fmt.Errorf("no supported languages can import configuration files")
		}
	}
	res := importer.ImportRepos(language.ImportReposArgs{
		Config: c,
		Path:   uc.repoFilePath,
		Prune:  uc.pruneRules,
		Cache:  rc,
	})
	return res.Gen, res.Empty, res.Error
}
