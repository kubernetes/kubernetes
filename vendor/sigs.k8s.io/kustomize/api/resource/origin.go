// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resource

import (
	"path/filepath"
	"strings"

	"sigs.k8s.io/kustomize/api/internal/git"
)

// Origin retains information about where resources in the output
// of `kustomize build` originated from
type Origin struct {
	// Path is the path to the resource, rooted from the directory upon
	// which `kustomize build` was invoked
	Path string

	// Repo is the remote repository that the resource originated from if it is
	// not from a local file
	Repo string

	// Ref is the ref of the remote repository that the resource originated from
	// if it is not from a local file
	Ref string
}

// Copy returns a copy of origin
func (origin *Origin) Copy() Origin {
	return *origin
}

// Append returns a copy of origin with a path appended to it
func (origin *Origin) Append(path string) *Origin {
	originCopy := origin.Copy()
	repoSpec, err := git.NewRepoSpecFromUrl(path)
	if err == nil {
		originCopy.Repo = repoSpec.Host + repoSpec.OrgRepo
		absPath := repoSpec.AbsPath()
		path = absPath[strings.Index(absPath[1:], "/")+1:][1:]
		originCopy.Path = ""
		originCopy.Ref = repoSpec.Ref
	}
	originCopy.Path = filepath.Join(originCopy.Path, path)
	return &originCopy
}

// String returns a string version of origin
func (origin *Origin) String() string {
	var anno string
	anno = anno + "path: " + origin.Path + "\n"
	if origin.Repo != "" {
		anno = anno + "repo: " + origin.Repo + "\n"
	}
	if origin.Ref != "" {
		anno = anno + "ref: " + origin.Ref + "\n"
	}
	return anno
}
