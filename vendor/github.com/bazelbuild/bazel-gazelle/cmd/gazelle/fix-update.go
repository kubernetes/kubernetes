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
	"sort"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/config"
	gzflag "github.com/bazelbuild/bazel-gazelle/flag"
	"github.com/bazelbuild/bazel-gazelle/label"
	"github.com/bazelbuild/bazel-gazelle/language"
	"github.com/bazelbuild/bazel-gazelle/merger"
	"github.com/bazelbuild/bazel-gazelle/repo"
	"github.com/bazelbuild/bazel-gazelle/resolve"
	"github.com/bazelbuild/bazel-gazelle/rule"
	"github.com/bazelbuild/bazel-gazelle/walk"
)

// updateConfig holds configuration information needed to run the fix and
// update commands. This includes everything in config.Config, but it also
// includes some additional fields that aren't relevant to other packages.
type updateConfig struct {
	dirs           []string
	emit           emitFunc
	repos          []repo.Repo
	workspaceFiles []*rule.File
	walkMode       walk.Mode
	patchPath      string
	patchBuffer    bytes.Buffer
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
	mode           string
	recursive      bool
	knownImports   []string
	repoConfigPath string
}

func (ucr *updateConfigurer) RegisterFlags(fs *flag.FlagSet, cmd string, c *config.Config) {
	uc := &updateConfig{}
	c.Exts[updateName] = uc

	c.ShouldFix = cmd == "fix"

	fs.StringVar(&ucr.mode, "mode", "fix", "print: prints all of the updated BUILD files\n\tfix: rewrites all of the BUILD files in place\n\tdiff: computes the rewrite but then just does a diff")
	fs.BoolVar(&ucr.recursive, "r", true, "when true, gazelle will update subdirectories recursively")
	fs.StringVar(&uc.patchPath, "patch", "", "when set with -mode=diff, gazelle will write to a file instead of stdout")
	fs.Var(&gzflag.MultiFlag{Values: &ucr.knownImports}, "known_import", "import path for which external resolution is skipped (can specify multiple times)")
	fs.StringVar(&ucr.repoConfigPath, "repo_config", "", "file where Gazelle should load repository configuration. Defaults to WORKSPACE.")
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
	} else if c.IndexLibraries {
		uc.walkMode = walk.VisitAllUpdateDirsMode
	} else {
		uc.walkMode = walk.UpdateDirsMode
	}

	// Load the repo configuration file (WORKSPACE by default) to find out
	// names and prefixes of other go_repositories. This affects external
	// dependency resolution for Go.
	// TODO(jayconrod): Go-specific code should be moved to language/go.
	if ucr.repoConfigPath == "" {
		ucr.repoConfigPath = filepath.Join(c.RepoRoot, "WORKSPACE")
	}
	repoConfigFile, err := rule.LoadWorkspaceFile(ucr.repoConfigPath, "")
	if err != nil && !os.IsNotExist(err) {
		return err
	} else if err == nil {
		c.Repos, _, err = repo.ListRepositories(repoConfigFile)
		if err != nil {
			return err
		}
	}
	for _, imp := range ucr.knownImports {
		uc.repos = append(uc.repos, repo.Repo{
			Name:     label.ImportPathToBazelRepoName(imp),
			GoPrefix: imp,
		})
	}
	for _, r := range c.Repos {
		if r.Kind() == "go_repository" {
			uc.repos = append(uc.repos, repo.Repo{
				Name:     r.Name(),
				GoPrefix: r.AttrString("importpath"),
			})
		}
	}

	// If the repo configuration file is not WORKSPACE, also load WORKSPACE
	// and any declared macro files so we can apply fixes.
	workspacePath := filepath.Join(c.RepoRoot, "WORKSPACE")
	var workspace *rule.File
	if ucr.repoConfigPath == workspacePath {
		workspace = repoConfigFile
	} else {
		workspace, err = rule.LoadWorkspaceFile(workspacePath, "")
		if err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	if workspace != nil {
		c.RepoName = findWorkspaceName(workspace)
		_, repoFileMap, err := repo.ListRepositories(workspace)
		if err != nil {
			return err
		}
		seen := make(map[*rule.File]bool)
		for _, f := range repoFileMap {
			if !seen[f] {
				uc.workspaceFiles = append(uc.workspaceFiles, f)
				seen[f] = true
			}
		}
		sort.Slice(uc.workspaceFiles, func(i, j int) bool {
			return uc.workspaceFiles[i].Path < uc.workspaceFiles[j].Path
		})
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

	// c is the configuration for the directory with directives applied.
	c *config.Config

	// rules is a list of generated Go rules.
	rules []*rule.Rule

	// imports contains opaque import information for each rule in rules.
	imports []interface{}

	// empty is a list of empty Go rules that may be deleted.
	empty []*rule.Rule

	// file is the build file being processed.
	file *rule.File

	// mappedKinds are mapped kinds used during this visit.
	mappedKinds    []config.MappedKind
	mappedKindInfo map[string]rule.KindInfo
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

func runFixUpdate(cmd command, args []string) (err error) {
	cexts := make([]config.Configurer, 0, len(languages)+3)
	cexts = append(cexts,
		&config.CommonConfigurer{},
		&updateConfigurer{},
		&walk.Configurer{},
		&resolve.Configurer{})
	mrslv := newMetaResolver()
	kinds := make(map[string]rule.KindInfo)
	loads := genericLoads
	for _, lang := range languages {
		cexts = append(cexts, lang)
		for kind, info := range lang.Kinds() {
			mrslv.AddBuiltin(kind, lang)
			kinds[kind] = info
		}
		loads = append(loads, lang.Loads()...)
	}
	ruleIndex := resolve.NewRuleIndex(mrslv.Resolver)

	c, err := newFixUpdateConfiguration(cmd, args, cexts)
	if err != nil {
		return err
	}

	if err := fixRepoFiles(c, loads); err != nil {
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
			if c.IndexLibraries && f != nil {
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
		var imports []interface{}
		for _, l := range languages {
			res := l.GenerateRules(language.GenerateArgs{
				Config:       c,
				Dir:          dir,
				Rel:          rel,
				File:         f,
				Subdirs:      subdirs,
				RegularFiles: regularFiles,
				GenFiles:     genFiles,
				OtherEmpty:   empty,
				OtherGen:     gen})
			if len(res.Gen) != len(res.Imports) {
				log.Panicf("%s: language %s generated %d rules but returned %d imports", rel, l.Name(), len(res.Gen), len(res.Imports))
			}
			empty = append(empty, res.Empty...)
			gen = append(gen, res.Gen...)
			imports = append(imports, res.Imports...)
		}
		if f == nil && len(gen) == 0 {
			return
		}

		// Apply and record relevant kind mappings.
		var (
			mappedKinds    []config.MappedKind
			mappedKindInfo = make(map[string]rule.KindInfo)
		)
		for _, r := range gen {
			if repl, ok := c.KindMap[r.Kind()]; ok {
				mappedKindInfo[repl.KindName] = kinds[r.Kind()]
				mappedKinds = append(mappedKinds, repl)
				mrslv.MappedKind(rel, repl)
				r.SetKind(repl.KindName)
			}
		}

		// Insert or merge rules into the build file.
		if f == nil {
			f = rule.EmptyFile(filepath.Join(dir, c.DefaultBuildFileName()), rel)
			for _, r := range gen {
				r.Insert(f)
			}
		} else {
			merger.MergeFile(f, empty, gen, merger.PreResolve,
				unionKindInfoMaps(kinds, mappedKindInfo))
		}
		visits = append(visits, visitRecord{
			pkgRel:         rel,
			c:              c,
			rules:          gen,
			imports:        imports,
			empty:          empty,
			file:           f,
			mappedKinds:    mappedKinds,
			mappedKindInfo: mappedKindInfo,
		})

		// Add library rules to the dependency resolution table.
		if c.IndexLibraries {
			for _, r := range f.Rules {
				ruleIndex.AddRule(c, r, f)
			}
		}
	})

	// Finish building the index for dependency resolution.
	ruleIndex.Finish()

	// Resolve dependencies.
	rc, cleanupRc := repo.NewRemoteCache(uc.repos)
	defer func() {
		if cerr := cleanupRc(); err == nil && cerr != nil {
			err = cerr
		}
	}()
	for _, v := range visits {
		for i, r := range v.rules {
			from := label.New(c.RepoName, v.pkgRel, r.Name())
			mrslv.Resolver(r, v.pkgRel).Resolve(v.c, ruleIndex, rc, r, v.imports[i], from)
		}
		merger.MergeFile(v.file, v.empty, v.rules, merger.PostResolve,
			unionKindInfoMaps(kinds, v.mappedKindInfo))
	}

	// Emit merged files.
	var exit error
	for _, v := range visits {
		merger.FixLoads(v.file, applyKindMappings(v.mappedKinds, loads))
		if err := uc.emit(v.c, v.file); err != nil {
			if err == exitError {
				exit = err
			} else {
				log.Print(err)
			}
		}
	}
	if uc.patchPath != "" {
		if err := ioutil.WriteFile(uc.patchPath, uc.patchBuffer.Bytes(), 0666); err != nil {
			return err
		}
	}

	return exit
}

func newFixUpdateConfiguration(cmd command, args []string, cexts []config.Configurer) (*config.Config, error) {
	c := config.New()

	fs := flag.NewFlagSet("gazelle", flag.ContinueOnError)
	// Flag will call this on any parse error. Don't print usage unless
	// -h or -help were passed explicitly.
	fs.Usage = func() {}

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

func fixRepoFiles(c *config.Config, loads []rule.LoadInfo) error {
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

	for _, f := range uc.workspaceFiles {
		merger.FixLoads(f, loads)
		if f.Path == filepath.Join(c.RepoRoot, "WORKSPACE") {
			removeLegacyGoRepository(f)
			if err := merger.CheckGazelleLoaded(f); err != nil {
				return err
			}
		}
		if err := uc.emit(c, f); err != nil {
			return err
		}
	}
	return nil
}

// removeLegacyGoRepository removes loads of go_repository from
// @io_bazel_rules_go. FixLoads should be called after this; it will load from
// @bazel_gazelle.
func removeLegacyGoRepository(f *rule.File) {
	for _, l := range f.Loads {
		if l.Name() == "@io_bazel_rules_go//go:def.bzl" {
			l.Remove("go_repository")
			if l.IsEmpty() {
				l.Delete()
			}
		}
	}
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

func unionKindInfoMaps(a, b map[string]rule.KindInfo) map[string]rule.KindInfo {
	if len(a) == 0 {
		return b
	}
	if len(b) == 0 {
		return a
	}
	result := make(map[string]rule.KindInfo, len(a)+len(b))
	for _, m := range []map[string]rule.KindInfo{a, b} {
		for k, v := range m {
			result[k] = v
		}
	}
	return result
}

// applyKindMappings returns a copy of LoadInfo that includes c.KindMap.
func applyKindMappings(mappedKinds []config.MappedKind, loads []rule.LoadInfo) []rule.LoadInfo {
	if len(mappedKinds) == 0 {
		return loads
	}

	// Add new RuleInfos or replace existing ones with merged ones.
	mappedLoads := make([]rule.LoadInfo, len(loads))
	copy(mappedLoads, loads)
	for _, mappedKind := range mappedKinds {
		mappedLoads = appendOrMergeKindMapping(mappedLoads, mappedKind)
	}
	return mappedLoads
}

// appendOrMergeKindMapping adds LoadInfo for the given replacement.
func appendOrMergeKindMapping(mappedLoads []rule.LoadInfo, mappedKind config.MappedKind) []rule.LoadInfo {
	// If mappedKind.KindLoad already exists in the list, create a merged copy.
	for i, load := range mappedLoads {
		if load.Name == mappedKind.KindLoad {
			mappedLoads[i].Symbols = append(load.Symbols, mappedKind.KindName)
			return mappedLoads
		}
	}

	// Add a new LoadInfo.
	return append(mappedLoads, rule.LoadInfo{
		Name:    mappedKind.KindLoad,
		Symbols: []string{mappedKind.KindName},
	})
}
