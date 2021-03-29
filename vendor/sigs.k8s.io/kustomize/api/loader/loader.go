// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package loader has a data loading interface and various implementations.
package loader

import (
	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/git"
)

// NewLoader returns a Loader pointed at the given target.
// If the target is remote, the loader will be restricted
// to the root and below only.  If the target is local, the
// loader will have the restrictions passed in.  Regardless,
// if a local target attempts to transitively load remote bases,
// the remote bases will all be root-only restricted.
func NewLoader(
	lr LoadRestrictorFunc,
	target string, fSys filesys.FileSystem) (ifc.Loader, error) {
	repoSpec, err := git.NewRepoSpecFromUrl(target)
	if err == nil {
		// The target qualifies as a remote git target.
		return newLoaderAtGitClone(
			repoSpec, fSys, nil, git.ClonerUsingGitExec)
	}
	root, err := demandDirectoryRoot(fSys, target)
	if err != nil {
		return nil, err
	}
	return newLoaderAtConfirmedDir(
		lr, root, fSys, nil, git.ClonerUsingGitExec), nil
}
