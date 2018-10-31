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

// Package loader has a data loading interface and various implementations.
package loader

import (
	"fmt"
	"path/filepath"
	"sigs.k8s.io/kustomize/pkg/ifc"

	"sigs.k8s.io/kustomize/pkg/fs"
)

// NewLoader returns a Loader given a target
// The target can be a local disk directory or a github Url
func NewLoader(target, r string, fSys fs.FileSystem) (ifc.Loader, error) {
	if !isValidLoaderPath(target, r) {
		return nil, fmt.Errorf("Not valid path: root='%s', loc='%s'\n", r, target)
	}

	if !isLocalTarget(target, fSys) && isRepoUrl(target) {
		return newGithubLoader(target, fSys)
	}

	l := newFileLoaderAtRoot(r, fSys)
	if isRootLoaderPath(r) {
		absPath, err := filepath.Abs(target)
		if err != nil {
			return nil, err
		}
		target = absPath
	}

	if !l.IsAbsPath(l.root, target) {
		return nil, fmt.Errorf("Not abs path: l.root='%s', loc='%s'\n", l.root, target)
	}
	root, err := l.fullLocation(l.root, target)
	if err != nil {
		return nil, err
	}
	return newFileLoaderAtRoot(root, l.fSys), nil
}

func isValidLoaderPath(target, root string) bool {
	return target != "" || root != ""
}

func isRootLoaderPath(root string) bool {
	return root == ""
}

// isLocalTarget checks if a file exists in the filesystem
func isLocalTarget(s string, fs fs.FileSystem) bool {
	return fs.Exists(s)
}
