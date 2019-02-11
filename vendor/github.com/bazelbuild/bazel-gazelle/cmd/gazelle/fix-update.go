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
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	gzflag "github.com/bazelbuild/bazel-gazelle/internal/flag"
	"github.com/bazelbuild/bazel-gazelle/internal/label"
	"github.com/bazelbuild/bazel-gazelle/internal/merger"
	"github.com/bazelbuild/bazel-gazelle/internal/repos"
	"github.com/bazelbuild/bazel-gazelle/internal/resolve"
	"github.com/bazelbuild/bazel-gazelle/internal/rule"
	"github.com/bazelbuild/bazel-gazelle/internal/walk"
)

// updateConfig holds configuration information needed to run the fix and
// update commands. This includes everything in config.Config, but it also
// includes some additional fields that aren't relevant to other packages.
type updateConfig struct {
	dirs        []string
	emit        emitFunc
	repos       []repos.Repo
	useIndex    bool
	walkMode    walk.Mode
	patchPath   string
	patchBuffer bytes.Buffer
}

type emitFunc func(c *config.Config, f *rule.File) error

var modeFromName = map[string]emitFunc{
	"print": printFile,
	"fix":   fixFile,
	"diff":  diffFile,
}

const updateName = "_update"

func getUpdateConfig(c *config.Config) *updateConfig {
	return c.Exts[updateName].(*updateConfig)
}

type updateConfigurer struct {
	mode      string
	recursive bool
}

func (ucr *updateConfigurer) RegisterFlags(fs *flag.FlagSet, cmd string, c *config.Config) {
	uc := &updateConfig{}
	c.Exts[updateName] = uc

	c.ShouldFix = cmd == "fix"

	fs.StringVar(&ucr.mode, "mode", "fix", "print: prints all of the updated BUILD files\n\tfix: rewrites all of the BUILD files in place\n\tdiff: computes the rewrite but then just does a diff")
	fs.BoolVar(&uc.useIndex, "index", true, "when true, gazelle will build an index of libraries in the workspace for dependency resolution")
	fs.BoolVar(&ucr.recursive, "r", true, "when true, gazelle will update subdirectories recursively")
	fs.StringVar(&uc.patchPath, "patch", "", "when set with -mode=diff, gazelle will write to a file instead of stdout")
}

func (ucr *updateConfigurer) CheckFlags(fs *flag.FlagSet, c *config.Config) error {
	uc := getUpdateConfig(c)

	var ok bool
	uc.emit, ok = modeFromName[ucr.mode]
	if !ok {
		return fmt.Errorf("unrecognized emit mode: %q", ucr.mode)
	}
	if uc.patchPath != "" && ucr.mode != "diff" {
		return fmt.Errorf("-patch set but -mode is %s, not diff", ucr.mode)
	}

	dirs := fs.Args()
	if len(dirs) == 0 {
		dirs = []string{"."}
	}
	uc.dirs = make([]string, len(dirs))
	for i := range dirs {
		dir, err := filepath.Abs(dirs[i])
		if err != nil {
			return fmt.Errorf("%s: failed to find absolute path: %v", dirs[i], err)
		}
		dir, err = filepath.EvalSymlinks(dir)
		if err != nil {
			return fmt.Errorf("%s: failed to resolve symlinks: %v", dirs[i], err)
		}
		if !isDescendingDir(dir, c.RepoRoot) {
			return fmt.Errorf("dir %q is not a subdirectory of repo root %q", dir, c.RepoRoot)
		}
		uc.dirs[i] = dir
	}

	if ucr.recursive {
		uc.walkMode = walk.VisitAllUpdateSubdirsMode
	} else if uc.useIndex {
		uc.walkMode = walk.VisitAllUpdateDirsMode
	} else {
		uc.walkMode = walk.UpdateDirsMode
	}

	return nil
}

func (ucr *updateConfigurer) KnownDirectives() []string { return nil }

func (ucr *updateConfigurer) Configure(c *config.Config, rel string, f *rule.File) {}

// visitRecord stores information about about a directory visited with
// packages.Walk.
type visitRecord struct {
	// pkgRel is the slash-separated path to the visited directory, relative to
	// the repository root. "" for the repository root itself.
	pkgRel string

	// rules is a list of generated Go rules.
	rules []*rule.Rule

	// empty is a list of empty Go rules that may be deleted.
	empty []*rule.Rule

	// file is the build file being processed.
	file *rule.File
}

type byPkgRel []visitRecord

func (vs byPkgRel) Len() int           { return len(vs) }
func (vs byPkgRel) Less(i, j int) bool { return vs[i].pkgRel < vs[j].pkgRel }
func (vs byPkgRel) Swap(i, j int)      { vs[i], vs[j] = vs[j], vs[i] }

var genericLoads = []rule.LoadInfo{
	{
		Name:    "@bazel_gazelle//:def.bzl",
		Symbols: []string{"gazelle"},
	},
}

func runFixUpdate(cmd command, args []string) error {
	cexts := make([]config.Configurer, 0, len(languages)+3)
	cexts = append(cexts,
		&config.CommonConfigurer{},
		&updateConfigurer{},
		&walk.Configurer{},
		&resolve.Configurer{})
	kindToResolver := make(map[string]resolve.Resolver)
	kinds := make(map[string]rule.KindInfo)
	loads := genericLoads
	for _, lang := range languages {
		cexts = append(cexts, lang)
		for kind, info := range lang.Kinds() {
			kindToResolver[kind] = lang
			kinds[kind] = info
		}
		loads = append(loads, lang.Loads()...)
	}
	ruleIndex := resolve.NewRuleIndex(kindToResolver)

	c, err := newFixUpdateConfiguration(cmd, args, cexts, loads)
	if err != nil {
		return err
	}

	if cmd == fixCmd {
		// Only check the version when "fix" is run. Generated build files
		// frequently work with older version of rules_go, and we don't want to
		// nag too much since there's no way to disable this warning.
		checkRulesGoVersion(c.RepoRoot)
	}

	// Visit all directories in the repository.
	var visits []visitRecord
	uc := getUpdateConfig(c)
	walk.Walk(c, cexts, uc.dirs, uc.walkMode, func(dir, rel string, c *config.Config, update bool, f *rule.File, subdirs, regularFiles, genFiles []string) {
		// If this file is ignored or if Gazelle was not asked to update this
		// directory, just index the build file and move on.
		if !update {
			if uc.useIndex && f != nil {
				for _, r := range f.Rules {
					ruleIndex.AddRule(c, r, f)
				}
			}
			return
		}

		// Fix any problems in the file.
		if f != nil {
			for _, l := range languages {
				l.Fix(c, f)
			}
		}

		// Generate rules.
		var empty, gen []*rule.Rule
		for _, l := range languages {
			lempty, lgen := l.GenerateRules(c, dir, rel, f, subdirs, regularFiles, genFiles, empty, gen)
			empty = append(empty, lempty...)
			gen = append(gen, lgen...)
		}
		if f == nil && len(gen) == 0 {
			return
		}

		// Insert or merge rules into the build file.
		if f == nil {
			f = rule.EmptyFile(filepath.Join(dir, c.DefaultBuildFileName()), rel)
			for _, r := range gen {
				r.Insert(f)
			}
		} else {
			merger.MergeFile(f, empty, gen, merger.PreResolve, kinds)
		}
		visits = append(visits, visitRecord{
			pkgRel: rel,
			rules:  gen,
			empty:  empty,
			file:   f,
		})

		// Add library rules to the dependency resolution table.
		for _, r := range f.Rules {
			ruleIndex.AddRule(c, r, f)
		}
	})

	// Finish building the index for dependency resolution.
	ruleIndex.Finish()

	// Resolve dependencies.
	rc := repos.NewRemoteCache(uc.repos)
	for _, v := range visits {
		for _, r := range v.rules {
			from := label.New(c.RepoName, v.pkgRel, r.Name())
			kindToResolver[r.Kind()].Resolve(c, ruleIndex, rc, r, from)
		}
		merger.MergeFile(v.file, v.empty, v.rules, merger.PostResolve, kinds)
	}

	// Emit merged files.
	for _, v := range visits {
		merger.FixLoads(v.file, loads)
		if err := uc.emit(c, v.file); err != nil {
			log.Print(err)
		}
	}
	if uc.patchPath != "" {
		if err := ioutil.WriteFile(uc.patchPath, uc.patchBuffer.Bytes(), 0666); err != nil {
			return err
		}
	}

	return nil
}

func newFixUpdateConfiguration(cmd command, args []string, cexts []config.Configurer, loads []rule.LoadInfo) (*config.Config, error) {
	c := config.New()

	fs := flag.NewFlagSet("gazelle", flag.ContinueOnError)
	// Flag will call this on any parse error. Don't print usage unless
	// -h or -help were passed explicitly.
	fs.Usage = func() {}

	var knownImports []string
	fs.Var(&gzflag.MultiFlag{Values: &knownImports}, "known_import", "import path for which external resolution is skipped (can specify multiple times)")

	for _, cext := range cexts {
		cext.RegisterFlags(fs, cmd.String(), c)
	}

	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			fixUpdateUsage(fs)
			return nil, err
		}
		// flag already prints the error; don't print it again.
		log.Fatal("Try -help for more information.")
	}

	for _, cext := range cexts {
		if err := cext.CheckFlags(fs, c); err != nil {
			return nil, err
		}
	}

	uc := getUpdateConfig(c)
	workspacePath := filepath.Join(c.RepoRoot, "WORKSPACE")
	if workspace, err := rule.LoadFile(workspacePath, ""); err != nil {
		if !os.IsNotExist(err) {
			return nil, err
		}
	} else {
		if err := fixWorkspace(c, workspace, loads); err != nil {
			return nil, err
		}
		c.RepoName = findWorkspaceName(workspace)
		uc.repos = repos.ListRepositories(workspace)
	}
	repoPrefixes := make(map[string]bool)
	for _, r := range uc.repos {
		repoPrefixes[r.GoPrefix] = true
	}
	for _, imp := range knownImports {
		if repoPrefixes[imp] {
			continue
		}
		repo := repos.Repo{
			Name:     label.ImportPathToBazelRepoName(imp),
			GoPrefix: imp,
		}
		uc.repos = append(uc.repos, repo)
	}

	return c, nil
}

func fixUpdateUsage(fs *flag.FlagSet) {
	fmt.Fprint(os.Stderr, `usage: gazelle [fix|update] [flags...] [package-dirs...]

The update command creates new build files and update existing BUILD files
when needed.

The fix command also creates and updates build files, and in addition, it may
make potentially breaking updates to usage of rules. For example, it may
delete obsolete rules or rename existing rules.

There are several output modes which can be selected with the -mode flag. The
output mode determines what Gazelle does with updated BUILD files.

  fix (default) - write updated BUILD files back to disk.
  print - print updated BUILD files to stdout.
  diff - diff updated BUILD files against existing files in unified format.

Gazelle accepts a list of paths to Go package directories to process (defaults
to the working directory if none are given). It recursively traverses
subdirectories. All directories must be under the directory specified by
-repo_root; if -repo_root is not given, this is the directory containing the
WORKSPACE file.

FLAGS:

`)
	fs.PrintDefaults()
}

func fixWorkspace(c *config.Config, workspace *rule.File, loads []rule.LoadInfo) error {
	uc := getUpdateConfig(c)
	if !c.ShouldFix {
		return nil
	}
	shouldFix := false
	for _, d := range uc.dirs {
		if d == c.RepoRoot {
			shouldFix = true
		}
	}
	if !shouldFix {
		return nil
	}

	merger.FixWorkspace(workspace)
	merger.FixLoads(workspace, loads)
	if err := merger.CheckGazelleLoaded(workspace); err != nil {
		return err
	}
	return uc.emit(c, workspace)
}

func findWorkspaceName(f *rule.File) string {
	for _, r := range f.Rules {
		if r.Kind() == "workspace" {
			return r.Name()
		}
	}
	return ""
}

func isDescendingDir(dir, root string) bool {
	rel, err := filepath.Rel(root, dir)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	return !strings.HasPrefix(rel, "..")
}

func findOutputPath(c *config.Config, f *rule.File) string {
	if c.ReadBuildFilesDir == "" && c.WriteBuildFilesDir == "" {
		return f.Path
	}
	baseDir := c.WriteBuildFilesDir
	if c.WriteBuildFilesDir == "" {
		baseDir = c.RepoRoot
	}
	outputDir := filepath.Join(baseDir, filepath.FromSlash(f.Pkg))
	defaultOutputPath := filepath.Join(outputDir, c.DefaultBuildFileName())
	files, err := ioutil.ReadDir(outputDir)
	if err != nil {
		// Ignore error. Directory probably doesn't exist.
		return defaultOutputPath
	}
	outputPath := rule.MatchBuildFileName(outputDir, c.ValidBuildFileNames, files)
	if outputPath == "" {
		return defaultOutputPath
	}
	return outputPath
}
