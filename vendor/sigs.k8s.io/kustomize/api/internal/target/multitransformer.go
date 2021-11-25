// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package target

import (
	"sigs.k8s.io/kustomize/api/resmap"
)

// multiTransformer contains a list of transformers.
type multiTransformer struct {
	transformers []resmap.Transformer
}

var _ resmap.Transformer = &multiTransformer{}

// newMultiTransformer constructs a multiTransformer.
func newMultiTransformer(t []resmap.Transformer) resmap.Transformer {
	r := &multiTransformer{
		transformers: make([]resmap.Transformer, len(t)),
	}
	copy(r.transformers, t)
	return r
}

// Transform applies the member transformers in order to the resources,
// optionally detecting and erroring on commutation conflict.
func (o *multiTransformer) Transform(m resmap.ResMap) error {
	for _, t := range o.transformers {
		if err := t.Transform(m); err != nil {
			return err
		}
		m.DropEmpties()
	}
	return nil
}
