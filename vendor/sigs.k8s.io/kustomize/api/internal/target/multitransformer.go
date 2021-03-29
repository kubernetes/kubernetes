// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package target

import (
	"fmt"

	"sigs.k8s.io/kustomize/api/resmap"
)

// multiTransformer contains a list of transformers.
type multiTransformer struct {
	transformers         []resmap.Transformer
	checkConflictEnabled bool
}

var _ resmap.Transformer = &multiTransformer{}

// newMultiTransformer constructs a multiTransformer.
func newMultiTransformer(t []resmap.Transformer) resmap.Transformer {
	r := &multiTransformer{
		transformers:         make([]resmap.Transformer, len(t)),
		checkConflictEnabled: false}
	copy(r.transformers, t)
	return r
}

// Transform applies the member transformers in order to the resources,
// optionally detecting and erroring on commutation conflict.
func (o *multiTransformer) Transform(m resmap.ResMap) error {
	if o.checkConflictEnabled {
		return o.transformWithCheckConflict(m)
	}
	return o.transform(m)
}

func (o *multiTransformer) transform(m resmap.ResMap) error {
	for _, t := range o.transformers {
		err := t.Transform(m)
		if err != nil {
			return err
		}
	}
	for _, r := range m.Resources() {
		empty, err := r.IsEmpty()
		if err != nil {
			return err
		}
		if empty {
			err := m.Remove(r.CurId())
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// Of the len(o.transformers)! possible transformer orderings, compare to a reversed order.
// A spot check to perform when the transformations are supposed to be commutative.
// Fail if there's a difference in the result.
func (o *multiTransformer) transformWithCheckConflict(m resmap.ResMap) error {
	mcopy := m.DeepCopy()
	err := o.transform(m)
	if err != nil {
		return err
	}
	o.reverseTransformers()
	err = o.transform(mcopy)
	if err != nil {
		return err
	}
	err = m.ErrorIfNotEqualSets(mcopy)
	if err != nil {
		return fmt.Errorf("found conflict between different patches\n%v", err)
	}
	return nil
}

func (o *multiTransformer) reverseTransformers() {
	for i, j := 0, len(o.transformers)-1; i < j; i, j = i+1, j-1 {
		o.transformers[i], o.transformers[j] = o.transformers[j], o.transformers[i]
	}
}
