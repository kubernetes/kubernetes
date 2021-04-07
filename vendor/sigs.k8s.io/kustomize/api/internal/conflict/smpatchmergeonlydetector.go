// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package conflict

import (
	"sigs.k8s.io/kustomize/api/resource"
)

// smPatchMergeOnlyDetector ignores conflicts,
// but does real strategic merge patching.
// This is part of an effort to eliminate dependence on
// apimachinery package to allow kustomize integration
// into kubectl (#2506 and #1500)
type smPatchMergeOnlyDetector struct{}

var _ resource.ConflictDetector = &smPatchMergeOnlyDetector{}

func (c *smPatchMergeOnlyDetector) HasConflict(
	_, _ *resource.Resource) (bool, error) {
	return false, nil
}

// There's at least one case that doesn't work.  Suppose one has a
// Deployment with a volume with the bizarre "emptyDir: {}" entry.
// If you want to get rid of this entry via a patch containing
// the entry "emptyDir: null", then the following won't work,
// because null entries are eliminated.
func (c *smPatchMergeOnlyDetector) MergePatches(
	r, patch *resource.Resource) (*resource.Resource, error) {
	err := r.ApplySmPatch(patch)
	return r, err
}
