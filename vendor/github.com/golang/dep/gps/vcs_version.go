// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"strings"

	"github.com/Masterminds/vcs"
	"github.com/pkg/errors"
)

// VCSVersion returns the current project version for an absolute path.
func VCSVersion(path string) (Version, error) {
	repo, err := vcs.NewRepo("", path)
	if err != nil {
		return nil, errors.Wrapf(err, "creating new repo for root: %s", path)
	}

	ver, err := repo.Current()
	if err != nil {
		return nil, errors.Wrapf(err, "finding current branch/version for root: %s", path)
	}

	rev, err := repo.Version()
	if err != nil {
		return nil, errors.Wrapf(err, "getting repo version for root: %s", path)
	}

	// First look through tags.
	tags, err := repo.Tags()
	if err != nil {
		return nil, errors.Wrapf(err, "getting repo tags for root: %s", path)
	}
	// Try to match the current version to a tag.
	if contains(tags, ver) {
		// Assume semver if it starts with a v.
		if strings.HasPrefix(ver, "v") {
			return NewVersion(ver).Pair(Revision(rev)), nil
		}

		return nil, errors.Errorf("version for root %s does not start with a v: %q", path, ver)
	}

	// Look for the current branch.
	branches, err := repo.Branches()
	if err != nil {
		return nil, errors.Wrapf(err, "getting repo branch for root: %s")
	}
	// Try to match the current version to a branch.
	if contains(branches, ver) {
		return NewBranch(ver).Pair(Revision(rev)), nil
	}

	return Revision(rev), nil
}

// contains checks if a array of strings contains a value
func contains(a []string, b string) bool {
	for _, v := range a {
		if b == v {
			return true
		}
	}
	return false
}
