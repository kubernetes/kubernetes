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
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/label"
	"github.com/bazelbuild/bazel-gazelle/internal/merger"
	"github.com/bazelbuild/bazel-gazelle/internal/packages"
	"github.com/bazelbuild/bazel-gazelle/internal/repos"
	"github.com/bazelbuild/bazel-gazelle/internal/resolve"
	"github.com/bazelbuild/bazel-gazelle/internal/rules"
	"github.com/bazelbuild/bazel-gazelle/internal/wspace"
	bf "github.com/bazelbuild/buildtools/build"
)

// updateConfig holds configuration information needed to run the fix and
// update commands. This includes everything in config.Config, but it also
// includes some additional fields that aren't relevant to other packages.
type updateConfig struct {
	c                 *config.Config
	emit              emitFunc
	outDir, outSuffix string
	repos             []repos.Repo
}

type emitFunc func(*config.Config, *bf.File, string) error

var modeFromName = map[string]emitFunc{
	"print": printFile,
	"fix":   fixFile,
	"diff":  diffFile,
}

// visitRecord stores information about about a directory visited with
// packages.Walk.
type visitRecord struct {
	// pkgRel is the slash-separated path to the visited directory, relative to
	// the repository root. "" for the repository root itself.
	pkgRel string

	// rules is a list of generated Go rules.
	rules []bf.Expr

	// empty is a list of empty Go rules that may be deleted.
	empty []bf.Expr

	// file is the build file being processed.
	file *bf.File
}

type byPkgRel []visitRecord

func (vs byPkgRel) Len() int           { return len(vs) }
func (vs byPkgRel) Less(i, j int) bool { return vs[i].pkgRel < vs[j].pkgRel }
func (vs byPkgRel) Swap(i, j int)      { vs[i], vs[j] = vs[j], vs[i] }

func runFixUpdate(cmd command, args []string) error {
	uc, err := newFixUpdateConfiguration(cmd, args)
	if err != nil {
		return err
	}
	if cmd == fixCmd {
		// Only check the version when "fix" is run. Generated build files
		// frequently work with older version of rules_go, and we don't want to
		// nag too much since there's no way to disable this warning.
		checkRulesGoVersion(uc.c.RepoRoot)
	}

	l := label.NewLabeler(uc.c)
	ruleIndex := resolve.NewRuleIndex()

	var visits []visitRecord

	// Visit all directories in the repository.
	packages.Walk(uc.c, uc.c.RepoRoot, func(dir, rel string, c *config.Config, pkg *packages.Package, file *bf.File, isUpdateDir bool) {
		// If this file is ignored or if Gazelle was not asked to update this
		// directory, just index the build file and move on.
		if !isUpdateDir {
			if file != nil {
				ruleIndex.AddRulesFromFile(c, file)
			}
			return
		}

		// Fix any problems in the file.
		if file != nil {
			file = merger.FixFileMinor(c, file)
			fixedFile := merger.FixFile(c, file)
			if cmd == fixCmd {
				file = fixedFile
			} else if fixedFile != file {
				log.Printf("%s: warning: file contains rules whose structure is out of date. Consider running 'gazelle fix'.", file.Path)
			}
		}

		// If the file exists, but no Go code is present, create an empty package.
		// This lets us delete existing rules.
		if pkg == nil && file != nil {
			pkg = packages.EmptyPackage(c, dir, rel)
		}

		// Generate new rules and merge them into the existing file (if present).
		if pkg != nil {
			g := rules.NewGenerator(c, l, file)
			rules, empty, err := g.GenerateRules(pkg)
			if err != nil {
				log.Print(err)
				return
			}
			if file == nil {
				file = &bf.File{
					Path: filepath.Join(c.RepoRoot, filepath.FromSlash(rel), c.DefaultBuildFileName()),
					Stmt: rules,
				}
			} else {
				file, rules = merger.MergeFile(rules, empty, file, merger.PreResolveAttrs)
			}
			visits = append(visits, visitRecord{
				pkgRel: rel,
				rules:  rules,
				empty:  empty,
				file:   file,
			})
		}

		// Add library rules to the dependency resolution table.
		if file != nil {
			ruleIndex.AddRulesFromFile(c, file)
		}
	})

	// Finish building the index for dependency resolution.
	ruleIndex.Finish()

	// Resolve dependencies.
	rc := repos.NewRemoteCache(uc.repos)
	resolver := resolve.NewResolver(uc.c, l, ruleIndex, rc)
	for i := range visits {
		for j := range visits[i].rules {
			visits[i].rules[j] = resolver.ResolveRule(visits[i].rules[j], visits[i].pkgRel)
		}
		visits[i].file, _ = merger.MergeFile(visits[i].rules, visits[i].empty, visits[i].file, merger.PostResolveAttrs)
	}

	// Emit merged files.
	for _, v := range visits {
		rules.SortLabels(v.file)
		v.file = merger.FixLoads(v.file)
		bf.Rewrite(v.file, nil) // have buildifier 'format' our rules.

		path := v.file.Path
		if uc.outDir != "" {
			stem := filepath.Base(v.file.Path) + uc.outSuffix
			path = filepath.Join(uc.outDir, v.pkgRel, stem)
		}
		if err := uc.emit(uc.c, v.file, path); err != nil {
			log.Print(err)
		}
	}
	return nil
}

func newFixUpdateConfiguration(cmd command, args []string) (*updateConfig, error) {
	uc := &updateConfig{c: &config.Config{}}
	var err error

	fs := flag.NewFlagSet("gazelle", flag.ContinueOnError)
	// Flag will call this on any parse error. Don't print usage unless
	// -h or -help were passed explicitly.
	fs.Usage = func() {}

	knownImports := multiFlag{}
	buildFileName := fs.String("build_file_name", "BUILD.bazel,BUILD", "comma-separated list of valid build file names.\nThe first element of the list is the name of output build files to generate.")
	buildTags := fs.String("build_tags", "", "comma-separated list of build tags. If not specified, Gazelle will not\n\tfilter sources with build constraints.")
	external := fs.String("external", "external", "external: resolve external packages with go_repository\n\tvendored: resolve external packages as packages in vendor/")
	var goPrefix explicitFlag
	fs.Var(&goPrefix, "go_prefix", "prefix of import paths in the current workspace")
	repoRoot := fs.String("repo_root", "", "path to a directory which corresponds to go_prefix, otherwise gazelle searches for it.")
	fs.Var(&knownImports, "known_import", "import path for which external resolution is skipped (can specify multiple times)")
	mode := fs.String("mode", "fix", "print: prints all of the updated BUILD files\n\tfix: rewrites all of the BUILD files in place\n\tdiff: computes the rewrite but then just does a diff")
	outDir := fs.String("experimental_out_dir", "", "write build files to an alternate directory tree")
	outSuffix := fs.String("experimental_out_suffix", "", "extra suffix appended to build file names. Only used if -experimental_out_dir is also set.")
	var proto explicitFlag
	fs.Var(&proto, "proto", "default: generates new proto rules\n\tdisable: does not touch proto rules\n\tlegacy (deprecated): generates old proto rules")
	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			fixUpdateUsage(fs)
			os.Exit(0)
		}
		// flag already prints the error; don't print it again.
		log.Fatal("Try -help for more information.")
	}

	uc.c.Dirs = fs.Args()
	if len(uc.c.Dirs) == 0 {
		uc.c.Dirs = []string{"."}
	}
	for i := range uc.c.Dirs {
		uc.c.Dirs[i], err = filepath.Abs(uc.c.Dirs[i])
		if err != nil {
			return nil, err
		}
	}

	if *repoRoot != "" {
		uc.c.RepoRoot = *repoRoot
	} else if len(uc.c.Dirs) == 1 {
		uc.c.RepoRoot, err = wspace.Find(uc.c.Dirs[0])
		if err != nil {
			return nil, fmt.Errorf("-repo_root not specified, and WORKSPACE cannot be found: %v", err)
		}
	} else {
		uc.c.RepoRoot, err = wspace.Find(".")
		if err != nil {
			return nil, fmt.Errorf("-repo_root not specified, and WORKSPACE cannot be found: %v", err)
		}
	}
	uc.c.RepoRoot, err = filepath.EvalSymlinks(uc.c.RepoRoot)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate symlinks for repo root: %v", err)
	}

	for _, dir := range uc.c.Dirs {
		if !isDescendingDir(dir, uc.c.RepoRoot) {
			return nil, fmt.Errorf("dir %q is not a subdirectory of repo root %q", dir, uc.c.RepoRoot)
		}
	}

	uc.c.ValidBuildFileNames = strings.Split(*buildFileName, ",")
	if len(uc.c.ValidBuildFileNames) == 0 {
		return nil, fmt.Errorf("no valid build file names specified")
	}

	uc.c.SetBuildTags(*buildTags)
	uc.c.PreprocessTags()

	if goPrefix.set {
		uc.c.GoPrefix = goPrefix.value
	} else {
		uc.c.GoPrefix, err = loadGoPrefix(uc.c)
		if err != nil {
			return nil, err
		}
	}
	if err := config.CheckPrefix(uc.c.GoPrefix); err != nil {
		return nil, err
	}

	uc.c.ShouldFix = cmd == fixCmd

	uc.c.DepMode, err = config.DependencyModeFromString(*external)
	if err != nil {
		return nil, err
	}

	if proto.set {
		uc.c.ProtoMode, err = config.ProtoModeFromString(proto.value)
		if err != nil {
			return nil, err
		}
		uc.c.ProtoModeExplicit = true
	}

	emit, ok := modeFromName[*mode]
	if !ok {
		return nil, fmt.Errorf("unrecognized emit mode: %q", *mode)
	}
	uc.emit = emit

	uc.outDir = *outDir
	uc.outSuffix = *outSuffix

	workspacePath := filepath.Join(uc.c.RepoRoot, "WORKSPACE")
	workspaceContent, err := ioutil.ReadFile(workspacePath)
	if os.IsNotExist(err) {
		workspaceContent = nil
	} else if err != nil {
		return nil, err
	}
	workspace, err := bf.Parse(workspacePath, workspaceContent)
	if err != nil {
		return nil, err
	}
	uc.repos = repos.ListRepositories(workspace)
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

	return uc, nil
}

func fixUpdateUsage(fs *flag.FlagSet) {
	fmt.Fprintln(os.Stderr, `usage: gazelle [fix|update] [flags...] [package-dirs...]

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

func loadBuildFile(c *config.Config, dir string) (*bf.File, error) {
	var buildPath string
	for _, base := range c.ValidBuildFileNames {
		p := filepath.Join(dir, base)
		fi, err := os.Stat(p)
		if err == nil {
			if fi.Mode().IsRegular() {
				buildPath = p
				break
			}
			continue
		}
		if !os.IsNotExist(err) {
			return nil, err
		}
	}
	if buildPath == "" {
		return nil, os.ErrNotExist
	}

	data, err := ioutil.ReadFile(buildPath)
	if err != nil {
		return nil, err
	}
	return bf.Parse(buildPath, data)
}

func loadGoPrefix(c *config.Config) (string, error) {
	f, err := loadBuildFile(c, c.RepoRoot)
	if err != nil {
		return "", errors.New("-go_prefix not set")
	}
	for _, d := range config.ParseDirectives(f) {
		if d.Key == "prefix" {
			return d.Value, nil
		}
	}
	for _, s := range f.Stmt {
		c, ok := s.(*bf.CallExpr)
		if !ok {
			continue
		}
		l, ok := c.X.(*bf.LiteralExpr)
		if !ok {
			continue
		}
		if l.Token != "go_prefix" {
			continue
		}
		if len(c.List) != 1 {
			return "", fmt.Errorf("-go_prefix not set, and %s has go_prefix(%v) with too many args", f.Path, c.List)
		}
		v, ok := c.List[0].(*bf.StringExpr)
		if !ok {
			return "", fmt.Errorf("-go_prefix not set, and %s has go_prefix(%v) which is not a string", f.Path, bf.FormatString(c.List[0]))
		}
		return v.Value, nil
	}
	return "", fmt.Errorf("-go_prefix not set, and no # gazelle:prefix directive found in %s", f.Path)
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
