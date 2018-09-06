/* Copyright 2018 The Bazel Authors. All rights reserved.

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

package walk

import (
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/pathtools"
	"github.com/bazelbuild/bazel-gazelle/internal/rule"
)

// WalkFunc is a callback called by Walk in each visited directory.
//
// dir is the absolute file system path to the directory being visited.
//
// rel is the relative slash-separated path to the directory from the
// repository root. Will be "" for the repository root directory itself.
//
// c is the configuration for the current directory. This may have been
// modified by directives in the directory's build file.
//
// update is true when the build file may be updated.
//
// f is the existing build file in the directory. Will be nil if there
// was no file.
//
// subdirs is a list of base names of subdirectories within dir, not
// including excluded files.
//
// regularFiles is a list of base names of regular files within dir, not
// including excluded files.
//
// genFiles is a list of names of generated files, found by reading
// "out" and "outs" attributes of rules in f.
type WalkFunc func(dir, rel string, c *config.Config, update bool, f *rule.File, subdirs, regularFiles, genFiles []string)

// Walk traverses the directory tree rooted at c.RepoRoot in depth-first order.
//
// Walk calls the Configure method on each configuration extension in cexts
// in each directory in pre-order, whether a build file is present in the
// directory or not.
//
// Walk calls the callback wf in post-order.
func Walk(c *config.Config, cexts []config.Configurer, wf WalkFunc) {
	cexts = append(cexts, &walkConfigurer{})
	knownDirectives := make(map[string]bool)
	for _, cext := range cexts {
		for _, d := range cext.KnownDirectives() {
			knownDirectives[d] = true
		}
	}

	updateRels := buildUpdateRels(c.RepoRoot, c.Dirs)
	symlinks := symlinkResolver{root: c.RepoRoot, visited: []string{c.RepoRoot}}

	var visit func(*config.Config, string, string, bool)
	visit = func(c *config.Config, dir, rel string, isUpdateDir bool) {
		haveError := false

		if !isUpdateDir {
			isUpdateDir = shouldUpdateDir(rel, updateRels)
		}

		// TODO: OPT: ReadDir stats all the files, which is slow. We just care about
		// names and modes, so we should use something like
		// golang.org/x/tools/internal/fastwalk to speed this up.
		files, err := ioutil.ReadDir(dir)
		if err != nil {
			log.Print(err)
			return
		}

		f, err := loadBuildFile(c, rel, dir, files)
		if err != nil {
			log.Print(err)
			haveError = true
		}

		c = configure(cexts, knownDirectives, c, rel, f)
		wc := getWalkConfig(c)

		var subdirs, regularFiles []string
		for _, fi := range files {
			base := fi.Name()
			switch {
			case base == "" || base[0] == '.' || base[0] == '_' || wc.isExcluded(base):
				continue

			case fi.IsDir() || fi.Mode()&os.ModeSymlink != 0 && symlinks.follow(dir, base):
				subdirs = append(subdirs, base)

			default:
				regularFiles = append(regularFiles, base)
			}
		}

		for _, sub := range subdirs {
			visit(c, filepath.Join(dir, sub), path.Join(rel, sub), isUpdateDir)
		}

		genFiles := findGenFiles(wc, f)
		update := !haveError && isUpdateDir && !wc.ignore
		wf(dir, rel, c, update, f, subdirs, regularFiles, genFiles)
	}
	visit(c, c.RepoRoot, "", false)
}

// buildUpdateRels builds a list of relative paths from the repository root
// directory (passed as an absolute file path) to directories that Gazelle
// may update. The relative paths are slash-separated. "" represents the
// root directory itself.
func buildUpdateRels(root string, dirs []string) []string {
	var updateRels []string
	for _, dir := range dirs {
		rel, err := filepath.Rel(root, dir)
		if err != nil {
			// This should have been verified when c was built.
			log.Panicf("%s: not a subdirectory of repository root %q", dir, root)
		}
		rel = filepath.ToSlash(rel)
		if rel == "." || rel == "/" {
			rel = ""
		}
		updateRels = append(updateRels, rel)
	}
	return updateRels
}

func shouldUpdateDir(rel string, updateRels []string) bool {
	for _, r := range updateRels {
		if pathtools.HasPrefix(rel, r) {
			return true
		}
	}
	return false
}

func loadBuildFile(c *config.Config, pkg, dir string, files []os.FileInfo) (*rule.File, error) {
	var err error
	readDir := dir
	readFiles := files
	if c.ReadBuildFilesDir != "" {
		readDir = filepath.Join(c.ReadBuildFilesDir, filepath.FromSlash(pkg))
		readFiles, err = ioutil.ReadDir(readDir)
		if err != nil {
			return nil, err
		}
	}
	path := rule.MatchBuildFileName(readDir, c.ValidBuildFileNames, readFiles)
	if path == "" {
		return nil, nil
	}
	return rule.LoadFile(path, pkg)
}

func configure(cexts []config.Configurer, knownDirectives map[string]bool, c *config.Config, rel string, f *rule.File) *config.Config {
	if rel != "" {
		c = c.Clone()
	}
	if f != nil {
		for _, d := range f.Directives {
			if !knownDirectives[d.Key] {
				log.Printf("%s: unknown directive: gazelle:%s", f.Path, d.Key)
			}
		}
	}
	for _, cext := range cexts {
		cext.Configure(c, rel, f)
	}
	return c
}

func findGenFiles(wc walkConfig, f *rule.File) []string {
	if f == nil {
		return nil
	}
	var strs []string
	for _, r := range f.Rules {
		for _, key := range []string{"out", "outs"} {
			if s := r.AttrString(key); s != "" {
				strs = append(strs, s)
			} else if ss := r.AttrStrings(key); len(ss) > 0 {
				strs = append(strs, ss...)
			}
		}
	}

	var genFiles []string
	for _, s := range strs {
		if !wc.isExcluded(s) {
			genFiles = append(genFiles, s)
		}
	}
	return genFiles
}

type symlinkResolver struct {
	root    string
	visited []string
}

// Decide if symlink dir/base should be followed.
func (r *symlinkResolver) follow(dir, base string) bool {
	if dir == r.root && strings.HasPrefix(base, "bazel-") {
		// Links such as bazel-<workspace>, bazel-out, bazel-genfiles are created by
		// Bazel to point to internal build directories.
		return false
	}
	// See if the symlink points to a tree that has been already visited.
	fullpath := filepath.Join(dir, base)
	dest, err := filepath.EvalSymlinks(fullpath)
	if err != nil {
		return false
	}
	if !filepath.IsAbs(dest) {
		dest, err = filepath.Abs(filepath.Join(dir, dest))
		if err != nil {
			return false
		}
	}
	for _, p := range r.visited {
		if pathtools.HasPrefix(dest, p) || pathtools.HasPrefix(p, dest) {
			return false
		}
	}
	r.visited = append(r.visited, dest)
	stat, err := os.Stat(fullpath)
	if err != nil {
		return false
	}
	return stat.IsDir()
}
