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
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/git"
	"sigs.k8s.io/kustomize/pkg/ifc"
)

// NewLoader returns a Loader.
func NewLoader(path string, fSys fs.FileSystem) (ifc.Loader, error) {
	repoSpec, err := git.NewRepoSpecFromUrl(path)
	if err == nil {
		return newLoaderAtGitClone(
			repoSpec, fSys, nil, git.ClonerUsingGitExec)
	}
	root, err := demandDirectoryRoot(fSys, path)
	if err != nil {
		return nil, err
	}
	return newLoaderAtConfirmedDir(
		root, fSys, nil, git.ClonerUsingGitExec), nil
}
