// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kio

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/monochromegane/go-gitignore"
	"sigs.k8s.io/kustomize/kyaml/ext"
)

// ignoreFilesMatcher handles `.krmignore` files, which allows for ignoring
// files or folders in a package. The format of this file is a subset of the
// gitignore format, with recursive patterns (like a/**/c) not supported. If a
// file or folder matches any of the patterns in the .krmignore file for the
// package, it will be excluded.
//
// It works as follows:
//
// * It will look for .krmignore file in the top folder and on the top level
//   of any subpackages. Subpackages are defined by the presence of a Krmfile
//   in the folder.
// * `.krmignore` files only cover files and folders for the package in which
//   it is defined. So ignore patterns defined in a parent package does not
//   affect which files are ignored from a subpackage.
// * An ignore pattern can not ignore a subpackage. So even if the parent
//   package contains a pattern that ignores the directory foo, if foo is a
//   subpackage, it will still be included if the IncludeSubpackages property
//   is set to true
type ignoreFilesMatcher struct {
	matchers []matcher
}

// readIgnoreFile checks whether there is a .krmignore file in the path, and
// if it is, reads it in and turns it into a matcher. If we can't find a file,
// we just add a matcher that match nothing.
func (i *ignoreFilesMatcher) readIgnoreFile(path string) error {
	i.verifyPath(path)
	m, err := gitignore.NewGitIgnore(filepath.Join(path, ext.IgnoreFileName()))
	if err != nil {
		if os.IsNotExist(err) {
			i.matchers = append(i.matchers, matcher{
				matcher:  gitignore.DummyIgnoreMatcher(false),
				basePath: path,
			})
			return nil
		}
		return err
	}
	i.matchers = append(i.matchers, matcher{
		matcher:  m,
		basePath: path,
	})
	return nil
}

// verifyPath checks whether the top matcher on the stack
// is correct for the provided filepath. Matchers are removed once
// we encounter a filepath that is not a subpath of the basepath for
// the matcher.
func (i *ignoreFilesMatcher) verifyPath(path string) {
	for j := len(i.matchers) - 1; j >= 0; j-- {
		matcher := i.matchers[j]
		if strings.HasPrefix(path, matcher.basePath) || path == matcher.basePath {
			i.matchers = i.matchers[:j+1]
			return
		}
	}
}

// matchFile checks whether the file given by the provided path matches
// any of the patterns in the .krmignore file for the package.
func (i *ignoreFilesMatcher) matchFile(path string) bool {
	if len(i.matchers) == 0 {
		return false
	}
	i.verifyPath(filepath.Dir(path))
	return i.matchers[len(i.matchers)-1].matcher.Match(path, false)
}

// matchFile checks whether the directory given by the provided path matches
// any of the patterns in the .krmignore file for the package.
func (i *ignoreFilesMatcher) matchDir(path string) bool {
	if len(i.matchers) == 0 {
		return false
	}
	i.verifyPath(path)
	return i.matchers[len(i.matchers)-1].matcher.Match(path, true)
}

// matcher wraps the gitignore matcher and the path to the folder
// where the file was found.
type matcher struct {
	matcher gitignore.IgnoreMatcher

	basePath string
}
