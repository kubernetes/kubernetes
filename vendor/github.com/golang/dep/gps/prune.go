// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golang/dep/internal/fs"
	"github.com/pkg/errors"
)

// PruneOptions represents the pruning options used to write the dependecy tree.
type PruneOptions uint8

const (
	// PruneNestedVendorDirs indicates if nested vendor directories should be pruned.
	PruneNestedVendorDirs PruneOptions = 1 << iota
	// PruneUnusedPackages indicates if unused Go packages should be pruned.
	PruneUnusedPackages
	// PruneNonGoFiles indicates if non-Go files should be pruned.
	// Files matching licenseFilePrefixes and legalFileSubstrings are kept in
	// an attempt to comply with legal requirements.
	PruneNonGoFiles
	// PruneGoTestFiles indicates if Go test files should be pruned.
	PruneGoTestFiles
)

// PruneOptionSet represents trinary distinctions for each of the types of
// prune rules (as expressed via PruneOptions): nested vendor directories,
// unused packages, non-go files, and go test files.
//
// The three-way distinction is between "none", "true", and "false", represented
// by uint8 values of 0, 1, and 2, respectively.
//
// This trinary distinction is necessary in order to record, with full fidelity,
// a cascading tree of pruning values, as expressed in CascadingPruneOptions; a
// simple boolean cannot delineate between "false" and "none".
type PruneOptionSet struct {
	NestedVendor   uint8
	UnusedPackages uint8
	NonGoFiles     uint8
	GoTests        uint8
}

// CascadingPruneOptions is a set of rules for pruning a dependency tree.
//
// The DefaultOptions are the global default pruning rules, expressed as a
// single PruneOptions bitfield. These global rules will cascade down to
// individual project rules, unless superseded.
type CascadingPruneOptions struct {
	DefaultOptions    PruneOptions
	PerProjectOptions map[ProjectRoot]PruneOptionSet
}

// PruneOptionsFor returns the PruneOptions bits for the given project,
// indicating which pruning rules should be applied to the project's code.
//
// It computes the cascade from default to project-specific options (if any) on
// the fly.
func (o CascadingPruneOptions) PruneOptionsFor(pr ProjectRoot) PruneOptions {
	po, has := o.PerProjectOptions[pr]
	if !has {
		return o.DefaultOptions
	}

	ops := o.DefaultOptions
	if po.NestedVendor != 0 {
		if po.NestedVendor == 1 {
			ops |= PruneNestedVendorDirs
		} else {
			ops &^= PruneNestedVendorDirs
		}
	}

	if po.UnusedPackages != 0 {
		if po.UnusedPackages == 1 {
			ops |= PruneUnusedPackages
		} else {
			ops &^= PruneUnusedPackages
		}
	}

	if po.NonGoFiles != 0 {
		if po.NonGoFiles == 1 {
			ops |= PruneNonGoFiles
		} else {
			ops &^= PruneNonGoFiles
		}
	}

	if po.GoTests != 0 {
		if po.GoTests == 1 {
			ops |= PruneGoTestFiles
		} else {
			ops &^= PruneGoTestFiles
		}
	}

	return ops
}

func defaultCascadingPruneOptions() CascadingPruneOptions {
	return CascadingPruneOptions{
		DefaultOptions:    PruneNestedVendorDirs,
		PerProjectOptions: map[ProjectRoot]PruneOptionSet{},
	}
}

var (
	// licenseFilePrefixes is a list of name prefixes for license files.
	licenseFilePrefixes = []string{
		"license",
		"licence",
		"copying",
		"unlicense",
		"copyright",
		"copyleft",
	}
	// legalFileSubstrings contains substrings that are likey part of a legal
	// declaration file.
	legalFileSubstrings = []string{
		"authors",
		"contributors",
		"legal",
		"notice",
		"disclaimer",
		"patent",
		"third-party",
		"thirdparty",
	}
)

// PruneProject remove excess files according to the options passed, from
// the lp directory in baseDir.
func PruneProject(baseDir string, lp LockedProject, options PruneOptions, logger *log.Logger) error {
	fsState, err := deriveFilesystemState(baseDir)

	if err != nil {
		return errors.Wrap(err, "could not derive filesystem state")
	}

	if (options & PruneNestedVendorDirs) != 0 {
		if err := pruneVendorDirs(fsState); err != nil {
			return errors.Wrapf(err, "failed to prune nested vendor directories")
		}
	}

	if (options & PruneUnusedPackages) != 0 {
		if _, err := pruneUnusedPackages(lp, fsState); err != nil {
			return errors.Wrap(err, "failed to prune unused packages")
		}
	}

	if (options & PruneNonGoFiles) != 0 {
		if err := pruneNonGoFiles(fsState); err != nil {
			return errors.Wrap(err, "failed to prune non-Go files")
		}
	}

	if (options & PruneGoTestFiles) != 0 {
		if err := pruneGoTestFiles(fsState); err != nil {
			return errors.Wrap(err, "failed to prune Go test files")
		}
	}

	// refresh fsState to figure out what's remaining
	fsState, err = deriveFilesystemState(baseDir)
	if err != nil {
		return errors.Wrap(err, "could not derive filesystem state")
	}

	if (options & PruneNonGoFiles) != 0 {
		if err := deleteLegaleseOnlyDirs(fsState); err != nil {
			return errors.Wrap(err, "failed to prune legalese-only dirs")
		}
	}

	if err := deleteEmptyDirs(fsState); err != nil {
		return errors.Wrap(err, "could not delete empty dirs")
	}

	return nil
}

// pruneVendorDirs deletes all nested vendor directories within baseDir.
func pruneVendorDirs(fsState filesystemState) error {
	for _, dir := range fsState.dirs {
		if filepath.Base(dir) == "vendor" {
			err := os.RemoveAll(filepath.Join(fsState.root, dir))
			if err != nil && !os.IsNotExist(err) {
				return err
			}
		}
	}

	for _, link := range fsState.links {
		if filepath.Base(link.path) == "vendor" {
			err := os.Remove(filepath.Join(fsState.root, link.path))
			if err != nil && !os.IsNotExist(err) {
				return err
			}
		}
	}

	return nil
}

// pruneUnusedPackages deletes unimported packages found in fsState.
// Determining whether packages are imported or not is based on the passed LockedProject.
func pruneUnusedPackages(lp LockedProject, fsState filesystemState) (map[string]interface{}, error) {
	unusedPackages := calculateUnusedPackages(lp, fsState)
	toDelete := collectUnusedPackagesFiles(fsState, unusedPackages)

	for _, path := range toDelete {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return nil, err
		}
	}

	return unusedPackages, nil
}

// calculateUnusedPackages generates a list of unused packages in lp.
func calculateUnusedPackages(lp LockedProject, fsState filesystemState) map[string]interface{} {
	unused := make(map[string]interface{})
	imported := make(map[string]interface{})

	for _, pkg := range lp.Packages() {
		imported[pkg] = nil
	}

	// Add the root package if it's not imported.
	if _, ok := imported["."]; !ok {
		unused["."] = nil
	}

	for _, dirPath := range fsState.dirs {
		pkg := filepath.ToSlash(dirPath)

		if _, ok := imported[pkg]; !ok {
			unused[pkg] = nil
		}
	}

	return unused
}

// collectUnusedPackagesFiles returns a slice of all files in the unused
// packages based on fsState.
func collectUnusedPackagesFiles(fsState filesystemState, unusedPackages map[string]interface{}) []string {
	// TODO(ibrasho): is this useful?
	files := make([]string, 0, len(unusedPackages))

	for _, path := range fsState.files {
		// Keep perserved files.
		if isPreservedFile(filepath.Base(path)) {
			continue
		}

		pkg := filepath.ToSlash(filepath.Dir(path))

		if _, ok := unusedPackages[pkg]; ok {
			files = append(files, filepath.Join(fsState.root, path))
		}
	}

	return files
}

func isSourceFile(path string) bool {
	ext := fileExt(path)

	// Refer to: https://github.com/golang/go/blob/release-branch.go1.9/src/go/build/build.go#L750
	switch ext {
	case ".go":
		return true
	case ".c":
		return true
	case ".cc", ".cpp", ".cxx":
		return true
	case ".m":
		return true
	case ".h", ".hh", ".hpp", ".hxx":
		return true
	case ".f", ".F", ".for", ".f90":
		return true
	case ".s":
		return true
	case ".S":
		return true
	case ".swig":
		return true
	case ".swigcxx":
		return true
	case ".syso":
		return true
	}
	return false
}

// pruneNonGoFiles delete all non-Go files existing in fsState.
//
// Files matching licenseFilePrefixes and legalFileSubstrings are not pruned.
func pruneNonGoFiles(fsState filesystemState) error {
	toDelete := make([]string, 0, len(fsState.files)/4)

	paths := fsState.files
	for _, link := range fsState.links {
		paths = append(paths, link.path)
	}

	for _, path := range paths {
		if isSourceFile(path) {
			continue
		}

		// Ignore perserved files.
		if isPreservedFile(filepath.Base(path)) {
			continue
		}

		toDelete = append(toDelete, filepath.Join(fsState.root, path))
	}

	for _, path := range toDelete {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}

	return nil
}

// isPreservedFile checks if the file name indicates that the file should be
// preserved based on licenseFilePrefixes or legalFileSubstrings.
// This applies only to non-source files.
func isPreservedFile(name string) bool {
	if isSourceFile(name) {
		return false
	}

	name = strings.ToLower(name)

	for _, prefix := range licenseFilePrefixes {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}

	for _, substring := range legalFileSubstrings {
		if strings.Contains(name, substring) {
			return true
		}
	}

	return false
}

// pruneGoTestFiles deletes all Go test files (*_test.go) in fsState.
func pruneGoTestFiles(fsState filesystemState) error {
	toDelete := make([]string, 0, len(fsState.files)/2)

	for _, path := range fsState.files {
		if strings.HasSuffix(path, "_test.go") {
			toDelete = append(toDelete, filepath.Join(fsState.root, path))
		}
	}

	for _, path := range toDelete {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}

	return nil
}

func isLegaleseOnly(fsState filesystemState, dir string) bool {
	for _, f := range fsState.files {
		if !strings.HasPrefix(f, dir) {
			continue
		}

		fname := filepath.Base(f)
		if !isPreservedFile(fname) {
			return false
		}
	}
	return true
}

func deleteLegaleseOnlyDirs(fsState filesystemState) error {
	sort.Sort(sort.Reverse(sort.StringSlice(fsState.dirs)))

	toDelete := make([]string, 0)

	for _, dir := range fsState.dirs {
		if isLegaleseOnly(fsState, dir) {
			toDelete = append(toDelete, filepath.Join(fsState.root, dir))
		}
	}

	for _, path := range toDelete {
		if err := os.RemoveAll(path); err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func deleteEmptyDirs(fsState filesystemState) error {
	sort.Sort(sort.Reverse(sort.StringSlice(fsState.dirs)))

	for _, dir := range fsState.dirs {
		path := filepath.Join(fsState.root, dir)

		notEmpty, err := fs.IsNonEmptyDir(path)
		if err != nil {
			return err
		}

		if !notEmpty {
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				return err
			}
		}
	}

	return nil
}

func fileExt(name string) string {
	i := strings.LastIndex(name, ".")
	if i < 0 {
		return ""
	}
	return name[i:]
}
