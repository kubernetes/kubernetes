/*
Copyright 2017 The Kubernetes Authors.

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

// Package value contains methods for assisting with transformation of values in storage.
package value

import (
	"bytes"
	"fmt"
	"sync"
	"time"
)

func init() {
	RegisterMetrics()
}

// Context is additional information that a storage transformation may need to verify the data at rest.
type Context interface {
	// AuthenticatedData should return an array of bytes that describes the current value. If the value changes,
	// the transformer may report the value as unreadable or tampered. This may be nil if no such description exists
	// or is needed. For additional verification, set this to data that strongly identifies the value, such as
	// the key and creation version of the stored data.
	AuthenticatedData() []byte
}

// Transformer allows a value to be transformed before being read from or written to the underlying store. The methods
// must be able to undo the transformation caused by the other.
type Transformer interface {
	// TransformFromStorage may transform the provided data from its underlying storage representation or return an error.
	// Stale is true if the object on disk is stale and a write to etcd should be issued, even if the contents of the object
	// have not changed.
	TransformFromStorage(data []byte, context Context) (out []byte, stale bool, err error)
	// TransformToStorage may transform the provided data into the appropriate form in storage or return an error.
	TransformToStorage(data []byte, context Context) (out []byte, err error)
}

type identityTransformer struct{}

// IdentityTransformer performs no transformation of the provided data.
var IdentityTransformer Transformer = identityTransformer{}

func (identityTransformer) TransformFromStorage(b []byte, ctx Context) ([]byte, bool, error) {
	return b, false, nil
}
func (identityTransformer) TransformToStorage(b []byte, ctx Context) ([]byte, error) {
	return b, nil
}

// DefaultContext is a simple implementation of Context for a slice of bytes.
type DefaultContext []byte

// AuthenticatedData returns itself.
func (c DefaultContext) AuthenticatedData() []byte { return []byte(c) }

// MutableTransformer allows a transformer to be changed safely at runtime.
type MutableTransformer struct {
	lock        sync.RWMutex
	transformer Transformer
}

// NewMutableTransformer creates a transformer that can be updated at any time by calling Set()
func NewMutableTransformer(transformer Transformer) *MutableTransformer {
	return &MutableTransformer{transformer: transformer}
}

// Set updates the nested transformer.
func (t *MutableTransformer) Set(transformer Transformer) {
	t.lock.Lock()
	t.transformer = transformer
	t.lock.Unlock()
}

func (t *MutableTransformer) TransformFromStorage(data []byte, context Context) (out []byte, stale bool, err error) {
	t.lock.RLock()
	transformer := t.transformer
	t.lock.RUnlock()
	return transformer.TransformFromStorage(data, context)
}
func (t *MutableTransformer) TransformToStorage(data []byte, context Context) (out []byte, err error) {
	t.lock.RLock()
	transformer := t.transformer
	t.lock.RUnlock()
	return transformer.TransformToStorage(data, context)
}

// PrefixTransformer holds a transformer interface and the prefix that the transformation is located under.
type PrefixTransformer struct {
	Prefix      []byte
	Transformer Transformer
}

type prefixTransformers struct {
	transformers []PrefixTransformer
	err          error
}

var _ Transformer = &prefixTransformers{}

// NewPrefixTransformers supports the Transformer interface by checking the incoming data against the provided
// prefixes in order. The first matching prefix will be used to transform the value (the prefix is stripped
// before the Transformer interface is invoked). The first provided transformer will be used when writing to
// the store.
func NewPrefixTransformers(err error, transformers ...PrefixTransformer) Transformer {
	if err == nil {
		err = fmt.Errorf("the provided value does not match any of the supported transformers")
	}
	return &prefixTransformers{
		transformers: transformers,
		err:          err,
	}
}

// TransformFromStorage finds the first transformer with a prefix matching the provided data and returns
// the result of transforming the value. It will always mark any transformation as stale that is not using
// the first transformer.
func (t *prefixTransformers) TransformFromStorage(data []byte, context Context) ([]byte, bool, error) {
	start := time.Now()
	for i, transformer := range t.transformers {
		if bytes.HasPrefix(data, transformer.Prefix) {
			result, stale, err := transformer.Transformer.TransformFromStorage(data[len(transformer.Prefix):], context)
			// To migrate away from encryption, user can specify an identity transformer higher up
			// (in the config file) than the encryption transformer. In that scenario, the identity transformer needs to
			// identify (during reads from disk) whether the data being read is encrypted or not. If the data is encrypted,
			// it shall throw an error, but that error should not prevent the next subsequent transformer from being tried.
			if len(transformer.Prefix) == 0 && err != nil {
				continue
			}
			if len(transformer.Prefix) == 0 {
				RecordTransformation("from_storage", "identity", start, err)
			} else {
				RecordTransformation("from_storage", string(transformer.Prefix), start, err)
			}
			return result, stale || i != 0, err
		}
	}
	RecordTransformation("from_storage", "unknown", start, t.err)
	return nil, false, t.err
}

// TransformToStorage uses the first transformer and adds its prefix to the data.
func (t *prefixTransformers) TransformToStorage(data []byte, context Context) ([]byte, error) {
	start := time.Now()
	transformer := t.transformers[0]
	prefixedData := make([]byte, len(transformer.Prefix), len(data)+len(transformer.Prefix))
	copy(prefixedData, transformer.Prefix)
	result, err := transformer.Transformer.TransformToStorage(data, context)
	RecordTransformation("to_storage", string(transformer.Prefix), start, err)
	if err != nil {
		return nil, err
	}
	prefixedData = append(prefixedData, result...)
	return prefixedData, nil
}
