/*
Copyright 2018 The Kubernetes Authors.

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

package loader

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"

	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
)

// TODO: 2018/Nov/20 remove this before next release.
// Leave only the true path.  Retaining only for
// quick revert.
const enforceRelocatable = true

// fileLoader loads files, returning an array of bytes.
// It also enforces two kustomization requirements:
//
//   1) relocatable
//
//      A kustomization and the resources, bases,
//      patches, etc. that it depends on should be
//      relocatable, so all path specifications
//      must be relative, not absolute.  The paths
//      are taken relative to the location of the
//      kusttomization file.
//
//   2) acyclic
//
//      There should be no cycles in overlay to base
//      relationships, including no cycles between
//      git repositories.
//
// The loader has a notion of a current working directory
// (CWD), called 'root', that is independent of the CWD
// of the process.  When `Load` is called with a file path
// argument, the load is done relative to this root,
// not relative to the process CWD.
//
// The loader offers a `New` method returning a new loader
// with a new root.  The new root can be one of two things,
// a remote git repo URL, or a directory specified relative
// to the current root.  In the former case, the remote
// repository is locally cloned, and the new loader is
// rooted on a path in that clone.
//
// Crucially, a root history is used to so that New fails
// if its argument either matches or is a parent of the
// current or any previously used root.
//
// This disallows:
//
//  * A base that is a git repository that, in turn,
//    specifies a base repository seen previously
//    in the loading process (a cycle).
//
//  * An overlay depending on a base positioned at or
//    above it.  I.e. '../foo' is OK, but '.', '..',
//    '../..', etc. are disallowed.  Allowing such a
//    base has no advantages and encourage cycles,
//    particularly if some future change were to
//    introduce globbing to file specifications in
//    the kustomization file.
//
type fileLoader struct {
	// Previously visited directories, tracked to
	// avoid cycles.  The last entry is the current root.
	roots []string
	// File system utilities.
	fSys fs.FileSystem
	// Used to clone repositories.
	cloner gitCloner
	// Used to clean up, as needed.
	cleaner func() error
}

// NewFileLoaderAtCwd returns a loader that loads from ".".
func NewFileLoaderAtCwd(fSys fs.FileSystem) *fileLoader {
	return newLoaderOrDie(fSys, ".")
}

// NewFileLoaderAtRoot returns a loader that loads from "/".
func NewFileLoaderAtRoot(fSys fs.FileSystem) *fileLoader {
	return newLoaderOrDie(fSys, "/")
}

// Root returns the absolute path that is prepended to any
// relative paths used in Load.
func (l *fileLoader) Root() string {
	return l.roots[len(l.roots)-1]
}

func newLoaderOrDie(fSys fs.FileSystem, path string) *fileLoader {
	l, err := newFileLoaderAt(
		path, fSys, []string{}, simpleGitCloner)
	if err != nil {
		log.Fatalf("unable to make loader at '%s'; %v", path, err)
	}
	return l
}

// newFileLoaderAt returns a new fileLoader with given root.
func newFileLoaderAt(
	root string, fSys fs.FileSystem,
	roots []string, cloner gitCloner) (*fileLoader, error) {
	if root == "" {
		return nil, fmt.Errorf(
			"loader root cannot be empty")
	}
	root, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf(
			"absolute path error in '%s' : %v", root, err)
	}
	if !fSys.IsDir(root) {
		return nil, fmt.Errorf("absolute root dir '%s' does not exist", root)
	}
	return &fileLoader{
		roots:   append(roots, root),
		fSys:    fSys,
		cloner:  cloner,
		cleaner: func() error { return nil },
	}, nil
}

// New returns a new Loader, rooted relative to current loader,
// or rooted in a temp directory holding a git repo clone.
func (l *fileLoader) New(root string) (ifc.Loader, error) {
	if root == "" {
		return nil, fmt.Errorf("new root cannot be empty")
	}
	if isRepoUrl(root) {
		if err := l.seenBefore(root); err != nil {
			return nil, err
		}
		return newGitLoader(root, l.fSys, l.roots, l.cloner)
	}
	if enforceRelocatable && filepath.IsAbs(root) {
		return nil, fmt.Errorf("new root '%s' cannot be absolute", root)
	}
	// Get absolute path to squeeze out "..", ".", etc.
	// to facilitate the seenBefore test.
	absRoot, err := filepath.Abs(filepath.Join(l.Root(), root))
	if err != nil {
		return nil, fmt.Errorf(
			"problem joining '%s' to '%s': %v", l.Root(), root, err)
	}
	if err := l.seenBefore(absRoot); err != nil {
		return nil, err
	}
	return newFileLoaderAt(absRoot, l.fSys, l.roots, l.cloner)
}

// newGitLoader returns a new Loader pinned to a temporary
// directory holding a cloned git repo.
func newGitLoader(
	root string, fSys fs.FileSystem,
	roots []string, cloner gitCloner) (ifc.Loader, error) {
	tmpDirForRepo, pathInRepo, err := cloner(root)
	if err != nil {
		return nil, err
	}
	trueRoot := filepath.Join(tmpDirForRepo, pathInRepo)
	if !fSys.IsDir(trueRoot) {
		return nil, fmt.Errorf(
			"something wrong cloning '%s'; unable to find '%s'",
			root, trueRoot)
	}
	return &fileLoader{
		roots:   append(roots, root, trueRoot),
		fSys:    fSys,
		cloner:  cloner,
		cleaner: func() error { return fSys.RemoveAll(tmpDirForRepo) },
	}, nil
}

// seenBefore tests whether the current or any previously
// visited root begins with the given path.
func (l *fileLoader) seenBefore(path string) error {
	for _, r := range l.roots {
		if strings.HasPrefix(r, path) {
			return fmt.Errorf(
				"cycle detected: new root '%s' contains previous root '%s'",
				path, r)
		}
	}
	return nil
}

// Load returns content of file at the given relative path.
func (l *fileLoader) Load(path string) ([]byte, error) {
	if filepath.IsAbs(path) {
		if enforceRelocatable {
			return nil, fmt.Errorf(
				"must use relative path; '%s' is absolute", path)
		}
	} else {
		path = filepath.Join(l.Root(), path)
	}
	return l.fSys.ReadFile(path)
}

// Cleanup runs the cleaner.
func (l *fileLoader) Cleanup() error {
	return l.cleaner()
}
