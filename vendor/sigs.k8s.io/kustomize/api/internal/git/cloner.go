// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package git

import (
	"sigs.k8s.io/kustomize/kyaml/filesys"
)

// Cloner is a function that can clone a git repo.
type Cloner func(repoSpec *RepoSpec) error

// ClonerUsingGitExec uses a local git install, as opposed
// to say, some remote API, to obtain a local clone of
// a remote repo.
func ClonerUsingGitExec(repoSpec *RepoSpec) error {
	r, err := newCmdRunner(repoSpec.Timeout)
	if err != nil {
		return err
	}
	repoSpec.Dir = r.dir
	if err = r.run("init"); err != nil {
		return err
	}
	// git relative submodule need origin, see https://github.com/kubernetes-sigs/kustomize/issues/5131
	if err = r.run("remote", "add", "origin", repoSpec.CloneSpec()); err != nil {
		return err
	}
	ref := "HEAD"
	if repoSpec.Ref != "" {
		ref = repoSpec.Ref
	}
	// we use repoSpec.CloneSpec() instead of origin because on error,
	// the prior prints the actual repo url for the user.
	if err = r.run("fetch", "--depth=1", repoSpec.CloneSpec(), ref); err != nil {
		return err
	}
	if err = r.run("checkout", "FETCH_HEAD"); err != nil {
		return err
	}
	if repoSpec.Submodules {
		return r.run("submodule", "update", "--init", "--recursive")
	}
	return nil
}

// DoNothingCloner returns a cloner that only sets
// cloneDir field in the repoSpec.  It's assumed that
// the cloneDir is associated with some fake filesystem
// used in a test.
func DoNothingCloner(dir filesys.ConfirmedDir) Cloner {
	return func(rs *RepoSpec) error {
		rs.Dir = dir
		return nil
	}
}
