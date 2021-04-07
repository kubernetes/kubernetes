// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package conflict

import (
	"sigs.k8s.io/kustomize/api/resid"
	"sigs.k8s.io/kustomize/api/resource"
)

type cdFactory struct{}

var _ resource.ConflictDetectorFactory = &cdFactory{}

// NewFactory returns a new conflict detector factory.
func NewFactory() resource.ConflictDetectorFactory {
	return &cdFactory{}
}

// New returns an instance of smPatchMergeOnlyDetector.
func (c cdFactory) New(_ resid.Gvk) (resource.ConflictDetector, error) {
	return &smPatchMergeOnlyDetector{}, nil
}
