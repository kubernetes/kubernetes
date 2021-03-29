// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package resource

import "sigs.k8s.io/kustomize/api/resid"

// ConflictDetector detects conflicts between resources.
type ConflictDetector interface {
	// HasConflict returns true if the given resources have a conflict.
	HasConflict(patch1, patch2 *Resource) (bool, error)
	// Merge two resources into one.
	MergePatches(patch1, patch2 *Resource) (*Resource, error)
}

// ConflictDetectorFactory makes instances of ConflictDetector that know
// how to handle the given Group, Version, Kind tuple.
type ConflictDetectorFactory interface {
	New(gvk resid.Gvk) (ConflictDetector, error)
}
