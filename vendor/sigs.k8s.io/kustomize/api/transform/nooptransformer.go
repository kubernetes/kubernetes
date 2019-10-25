// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package transform

import "sigs.k8s.io/kustomize/api/resmap"

// noOpTransformer contains a no-op transformer.
type noOpTransformer struct{}

var _ resmap.Transformer = &noOpTransformer{}

// newNoOpTransformer constructs a noOpTransformer.
func newNoOpTransformer() resmap.Transformer {
	return &noOpTransformer{}
}

// Transform does nothing.
func (o *noOpTransformer) Transform(_ resmap.ResMap) error {
	return nil
}
