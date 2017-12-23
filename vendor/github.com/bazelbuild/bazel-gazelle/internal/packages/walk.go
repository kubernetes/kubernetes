/* Copyright 2016 The Bazel Authors. All rights reserved.

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

package packages

import (
	"go/build"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	"github.com/bazelbuild/bazel-gazelle/internal/pathtools"
	bf "github.com/bazelbuild/buildtools/build"
)

// A WalkFunc is a callback called by Walk in each visited directory.
//
// dir is the absolute file system path to the directory being visited.
//
// rel is the relative slash-separated path to the directory from the
// repository root. Will be "" for the repository root directory itself.
//
// c is the configuration for the current directory. This may have been
// modified by directives in the directory's build file.
//
// pkg contains information about how to build source code in the directory.
// Will be nil for directories that don't contain buildable code, directories
// that Gazelle was not asked update, and directories where Walk
// encountered errors.
//
// oldFile is the existing build file in the directory. Will be nil if there
// was no file.
//
// isUpdateDir is true for directories that Gazelle was asked to update.
type WalkFunc func(dir, rel string, c *config.Config, pkg *Package, oldFile *bf.File, isUpdateDir bool)

// Walk traverses a directory tree. In each directory, Walk parses existing
// build files. In directories that Gazelle was asked to update (c.Dirs), Walk
// also parses source files and infers build information.
//
// c is the base configuration for the repository. c may be copied and modified
// by directives found in build files.
//
// root is an absolute file path to the directory to traverse.
//
// f is a function that will be called for each visited directory.
func Walk(c *config.Config, root string, f WalkFunc) {
	// Determine relative paths for the directories to be updated.
	var updateRels []string
	for _, dir := range c.Dirs {
		rel, err := filepath.Rel(c.RepoRoot, dir)
		if err != nil {
			// This should have been verified when c was built.
			log.Panicf("%s: not a subdirectory of repository root %q", dir, c.RepoRoot)
		}
		rel = filepath.ToSlash(rel)
		if rel == "." || rel == "/" {
			rel = ""
		}
		updateRels = append(updateRels, rel)
	}
	rootRel, err := filepath.Rel(c.RepoRoot, root)
	if err != nil {
		log.Panicf("%s: not a subdirectory of repository root %q", root, c.RepoRoot)
	}
	if rootRel == "." || rootRel == "/" {
		rootRel = ""
	}

	symlinks := symlinkResolver{root: root, visited: []string{root}}

	// visit walks the directory tree in post-order. It returns whether the
	// given directory or any subdirectory contained a build file or buildable
	// source code. This affects whether "testdata" directories are considered
	// data dependencies.
	var visit func(*config.Config, string, string, bool, []string) bool
	visit = func(c *config.Config, dir, rel string, isUpdateDir bool, excluded []string) bool {
		// Check if this directory should be updated.
		if !isUpdateDir {
			for _, updateRel := range updateRels {
				if pathtools.HasPrefix(rel, updateRel) {
					isUpdateDir = true
				}
			}
		}

		// Look for an existing BUILD file.
		var oldFile *bf.File
		haveError := false
		for _, base := range c.ValidBuildFileNames {
			oldPath := filepath.Join(dir, base)
			st, err := os.Stat(oldPath)
			if os.IsNotExist(err) || err == nil && st.IsDir() {
				continue
			}
			oldData, err := ioutil.ReadFile(oldPath)
			if err != nil {
				log.Print(err)
				haveError = true
				continue
			}
			if oldFile != nil {
				log.Printf("in directory %s, multiple Bazel files are present: %s, %s",
					dir, filepath.Base(oldFile.Path), base)
				haveError = true
				continue
			}
			oldFile, err = bf.Parse(oldPath, oldData)
			if err != nil {
				log.Print(err)
				haveError = true
				continue
			}
		}

		// Process directives in the build file. If this is a vendor directory,
		// set an empty prefix.
		if path.Base(rel) == "vendor" {
			cCopy := *c
			cCopy.GoPrefix = ""
			cCopy.GoPrefixRel = rel
			c = &cCopy
		}
		var directives []config.Directive
		if oldFile != nil {
			directives = config.ParseDirectives(oldFile)
			c = config.ApplyDirectives(c, directives, rel)
		}
		c = config.InferProtoMode(c, rel, oldFile, directives)

		var ignore bool
		for _, d := range directives {
			switch d.Key {
			case "exclude":
				excluded = append(excluded, d.Value)
			case "ignore":
				ignore = true
			}
		}

		// List files and subdirectories.
		files, err := ioutil.ReadDir(dir)
		if err != nil {
			log.Print(err)
			return false
		}
		if c.ProtoMode == config.DefaultProtoMode {
			excluded = append(excluded, findPbGoFiles(files, excluded)...)
		}

		var pkgFiles, otherFiles, subdirs []string
		for _, f := range files {
			base := f.Name()
			switch {
			case base == "" || base[0] == '.' || base[0] == '_' || isExcluded(excluded, base):
				continue

			case f.IsDir():
				subdirs = append(subdirs, base)

			case strings.HasSuffix(base, ".go") ||
				(c.ProtoMode != config.DisableProtoMode && strings.HasSuffix(base, ".proto")):
				pkgFiles = append(pkgFiles, base)

			case f.Mode()&os.ModeSymlink != 0 && symlinks.follow(dir, base):
				subdirs = append(subdirs, base)

			default:
				otherFiles = append(otherFiles, base)
			}
		}
		// Recurse into subdirectories.
		hasTestdata := false
		subdirHasPackage := false
		for _, sub := range subdirs {
			subdirExcluded := excludedForSubdir(excluded, sub)
			hasPackage := visit(c, filepath.Join(dir, sub), path.Join(rel, sub), isUpdateDir, subdirExcluded)
			if sub == "testdata" && !hasPackage {
				hasTestdata = true
			}
			subdirHasPackage = subdirHasPackage || hasPackage
		}

		hasPackage := subdirHasPackage || oldFile != nil
		if haveError || !isUpdateDir || ignore {
			f(dir, rel, c, nil, oldFile, false)
			return hasPackage
		}

		// Build a package from files in this directory.
		var genFiles []string
		if oldFile != nil {
			genFiles = findGenFiles(oldFile, excluded)
		}
		pkg := buildPackage(c, dir, rel, pkgFiles, otherFiles, genFiles, hasTestdata)
		f(dir, rel, c, pkg, oldFile, true)
		return hasPackage || pkg != nil
	}

	visit(c, root, rootRel, false, nil)
}

// buildPackage reads source files in a given directory and returns a Package
// containing information about those files and how to build them.
//
// If no buildable .go files are found in the directory, nil will be returned.
// If the directory contains multiple buildable packages, the package whose
// name matches the directory base name will be returned. If there is no such
// package or if an error occurs, an error will be logged, and nil will be
// returned.
func buildPackage(c *config.Config, dir, rel string, pkgFiles, otherFiles, genFiles []string, hasTestdata bool) *Package {
	// Process .go and .proto files first, since these determine the package name.
	packageMap := make(map[string]*packageBuilder)
	cgo := false
	var pkgFilesWithUnknownPackage []fileInfo
	for _, f := range pkgFiles {
		var info fileInfo
		switch path.Ext(f) {
		case ".go":
			info = goFileInfo(c, dir, rel, f)
		case ".proto":
			info = protoFileInfo(c, dir, rel, f)
		default:
			log.Panicf("file cannot determine package name: %s", f)
		}
		if info.packageName == "" {
			pkgFilesWithUnknownPackage = append(pkgFilesWithUnknownPackage, info)
			continue
		}
		if info.packageName == "documentation" {
			// go/build ignores this package
			continue
		}

		cgo = cgo || info.isCgo

		if _, ok := packageMap[info.packageName]; !ok {
			packageMap[info.packageName] = &packageBuilder{
				name:        info.packageName,
				dir:         dir,
				rel:         rel,
				hasTestdata: hasTestdata,
			}
		}
		if err := packageMap[info.packageName].addFile(c, info, false); err != nil {
			log.Print(err)
		}
	}

	// Select a package to generate rules for.
	pkg, err := selectPackage(c, dir, packageMap)
	if err != nil {
		if _, ok := err.(*build.NoGoError); !ok {
			log.Print(err)
		}
		return nil
	}

	// Add files with unknown packages. This happens when there are parse
	// or I/O errors. We should keep the file in the srcs list and let the
	// compiler deal with the error.
	for _, info := range pkgFilesWithUnknownPackage {
		if err := pkg.addFile(c, info, cgo); err != nil {
			log.Print(err)
		}
	}

	// Process the other static files.
	for _, file := range otherFiles {
		info := otherFileInfo(dir, rel, file)
		if err := pkg.addFile(c, info, cgo); err != nil {
			log.Print(err)
		}
	}

	// Process generated files. Note that generated files may have the same names
	// as static files. Bazel will use the generated files, but we will look at
	// the content of static files, assuming they will be the same.
	staticFiles := make(map[string]bool)
	for _, f := range pkgFiles {
		staticFiles[f] = true
	}
	for _, f := range otherFiles {
		staticFiles[f] = true
	}
	for _, f := range genFiles {
		if staticFiles[f] {
			continue
		}
		info := fileNameInfo(dir, rel, f)
		if err := pkg.addFile(c, info, cgo); err != nil {
			log.Print(err)
		}
	}

	if pkg.importPath == "" {
		if err := pkg.inferImportPath(c); err != nil {
			log.Print(err)
			return nil
		}
	}
	return pkg.build()
}

func selectPackage(c *config.Config, dir string, packageMap map[string]*packageBuilder) (*packageBuilder, error) {
	buildablePackages := make(map[string]*packageBuilder)
	for name, pkg := range packageMap {
		if pkg.isBuildable(c) {
			buildablePackages[name] = pkg
		}
	}

	if len(buildablePackages) == 0 {
		return nil, &build.NoGoError{Dir: dir}
	}

	if len(buildablePackages) == 1 {
		for _, pkg := range buildablePackages {
			return pkg, nil
		}
	}

	if pkg, ok := buildablePackages[defaultPackageName(c, dir)]; ok {
		return pkg, nil
	}

	err := &build.MultiplePackageError{Dir: dir}
	for name, pkg := range buildablePackages {
		// Add the first file for each package for the error message.
		// Error() method expects these lists to be the same length. File
		// lists must be non-empty. These lists are only created by
		// buildPackage for packages with .go files present.
		err.Packages = append(err.Packages, name)
		err.Files = append(err.Files, pkg.firstGoFile())
	}
	return nil, err
}

func defaultPackageName(c *config.Config, dir string) string {
	if dir != c.RepoRoot {
		return filepath.Base(dir)
	}
	name := path.Base(c.GoPrefix)
	if name == "." || name == "/" {
		// This can happen if go_prefix is empty or is all slashes.
		return "unnamed"
	}
	return name
}

func findGenFiles(f *bf.File, excluded []string) []string {
	var strs []string
	for _, r := range f.Rules("") {
		for _, key := range []string{"out", "outs"} {
			switch e := r.Attr(key).(type) {
			case *bf.StringExpr:
				strs = append(strs, e.Value)
			case *bf.ListExpr:
				for _, elem := range e.List {
					if s, ok := elem.(*bf.StringExpr); ok {
						strs = append(strs, s.Value)
					}
				}
			}
		}
	}

	var genFiles []string
	for _, s := range strs {
		if !isExcluded(excluded, s) {
			genFiles = append(genFiles, s)
		}
	}
	return genFiles
}

func findPbGoFiles(files []os.FileInfo, excluded []string) []string {
	var pbGoFiles []string
	for _, f := range files {
		name := f.Name()
		if strings.HasSuffix(name, ".proto") && !isExcluded(excluded, name) {
			pbGoFiles = append(pbGoFiles, name[:len(name)-len(".proto")]+".pb.go")
		}
	}
	return pbGoFiles
}

func isExcluded(excluded []string, base string) bool {
	for _, e := range excluded {
		if base == e {
			return true
		}
	}
	return false
}

func excludedForSubdir(excluded []string, subdir string) []string {
	var filtered []string
	for _, e := range excluded {
		i := strings.IndexByte(e, '/')
		if i < 0 || i == len(e)-1 || e[:i] != subdir {
			continue
		}
		filtered = append(filtered, e[i+1:])
	}
	return filtered
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
