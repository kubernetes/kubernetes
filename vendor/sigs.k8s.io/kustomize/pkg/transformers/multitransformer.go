/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package transformers

import (
	"fmt"
	"sigs.k8s.io/kustomize/pkg/resource"

	"sigs.k8s.io/kustomize/pkg/resmap"
)

// multiTransformer contains a list of transformers.
type multiTransformer struct {
	transformers         []Transformer
	checkConflictEnabled bool
	rf                   *resource.Factory
}

var _ Transformer = &multiTransformer{}

// NewMultiTransformer constructs a multiTransformer.
func NewMultiTransformer(t []Transformer) Transformer {
	r := &multiTransformer{
		transformers:         make([]Transformer, len(t)),
		checkConflictEnabled: false}
	copy(r.transformers, t)
	return r
}

// NewMultiTransformerWithConflictCheck constructs a multiTransformer with checking of conflicts.
func NewMultiTransformerWithConflictCheck(t []Transformer) Transformer {
	r := &multiTransformer{
		transformers:         make([]Transformer, len(t)),
		checkConflictEnabled: true}
	copy(r.transformers, t)
	return r
}

// Transform prepends the name prefix.
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
	return nil
}

// Of the len(o.transformers)! possible transformer orderings, compare to a reversed order.
// A spot check to perform when the transformations are supposed to be commutative.
// Fail if there's a difference in the result.
func (o *multiTransformer) transformWithCheckConflict(m resmap.ResMap) error {
	mcopy := m.DeepCopy(o.rf)
	err := o.transform(m)
	if err != nil {
		return err
	}
	o.reverseTransformers()
	err = o.transform(mcopy)
	if err != nil {
		return err
	}
	err = m.ErrorIfNotEqual(mcopy)
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
