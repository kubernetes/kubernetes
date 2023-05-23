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
	"context"
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/errors"
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
	TransformFromStorage(ctx context.Context, data []byte, dataCtx Context) (out []byte, stale bool, err error)
	// TransformToStorage may transform the provided data into the appropriate form in storage or return an error.
	TransformToStorage(ctx context.Context, data []byte, dataCtx Context) (out []byte, err error)
}

// DefaultContext is a simple implementation of Context for a slice of bytes.
type DefaultContext []byte

// AuthenticatedData returns itself.
func (c DefaultContext) AuthenticatedData() []byte { return c }

// PrefixTransformer holds a transformer interface and the prefix that the transformation is located under.
type PrefixTransformer struct {
	Prefix      []byte
	Transformer Transformer
}

type prefixTransformers struct {
	// The first transfom on the list of input transformers
	firstTransformer PrefixTransformer
	// The map of input transformers with prefix as key and a slice of transformers
	// with the same prefix with the original order
	transformers map[string][]PrefixTransformer
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

	tfm := map[string][]PrefixTransformer{}
	for _, transformer := range transformers {
		val, ok := tfm[string(transformer.Prefix)]
		if ok {
			tfm[string(transformer.Prefix)] = append(val, transformer)
		} else {
			var tfs []PrefixTransformer
			tfm[string(transformer.Prefix)] = append(tfs, transformer)
		}
	}

	return &prefixTransformers{
		firstTransformer: transformers[0],
		transformers:     tfm,
		err:              err,
	}
}

// TransformFromStorage finds the first transformer with a prefix matching the provided data and returns
// the result of transforming the value. It will always mark any transformation as stale that is not using
// the first transformer.
func (t *prefixTransformers) TransformFromStorage(ctx context.Context, data []byte, dataCtx Context) ([]byte, bool, error) {
	start := time.Now()
	var errs []error

	// Loop through all the transformers with the same prefix
	transformers := t.transformers[getDataPrefix(data)]
	for i, transformer := range transformers {
		result, stale, err := transformer.Transformer.TransformFromStorage(ctx, data[len(transformer.Prefix):], dataCtx)
		// To migrate away from encryption, user can specify an identity transformer higher up
		// (in the config file) than the encryption transformer. In that scenario, the identity transformer needs to
		// identify (during reads from disk) whether the data being read is encrypted or not. If the data is encrypted,
		// it shall throw an error, but that error should not prevent the next subsequent transformer from being tried.
		if len(transformer.Prefix) == 0 && err != nil {
			continue
		}
		if len(transformer.Prefix) == 0 {
			RecordTransformation("from_storage", "identity", time.Since(start), err)
		} else {
			RecordTransformation("from_storage", string(transformer.Prefix), time.Since(start), err)
		}

		// It is valid to have overlapping prefixes when the same encryption provider
		// is specified multiple times but with different keys (the first provider is
		// being rotated to and some later provider is being rotated away from).
		//
		// Example:
		//
		//  {
		//    "aescbc": {
		//      "keys": [
		//        {
		//          "name": "2",
		//          "secret": "some key 2"
		//        }
		//      ]
		//    }
		//  },
		//  {
		//    "aescbc": {
		//      "keys": [
		//        {
		//          "name": "1",
		//          "secret": "some key 1"
		//        }
		//      ]
		//    }
		//  },
		//
		// The transformers for both aescbc configs share the prefix k8s:enc:aescbc:v1:
		// but a failure in the first one should not prevent a later match from being attempted.
		// Thus we never short-circuit on a prefix match that results in an error.
		if err != nil {
			errs = append(errs, err)
			continue
		}

		// Detect if the transformation is stale by verifying if the current
		// transformer is the first transformer
		if string(transformer.Prefix) != string(t.firstTransformer.Prefix) || i != 0 {
			stale = true
		}

		return result, stale, err
	}

	if err := errors.Reduce(errors.NewAggregate(errs)); err != nil {
		return nil, false, err
	}
	RecordTransformation("from_storage", "unknown", time.Since(start), t.err)
	return nil, false, t.err
}

// TransformToStorage uses the first transformer and adds its prefix to the data.
func (t *prefixTransformers) TransformToStorage(ctx context.Context, data []byte, dataCtx Context) ([]byte, error) {
	start := time.Now()
	transformer := t.firstTransformer
	result, err := transformer.Transformer.TransformToStorage(ctx, data, dataCtx)
	RecordTransformation("to_storage", string(transformer.Prefix), time.Since(start), err)
	if err != nil {
		return nil, err
	}
	prefixedData := make([]byte, len(transformer.Prefix), len(result)+len(transformer.Prefix))
	copy(prefixedData, transformer.Prefix)
	prefixedData = append(prefixedData, result...)
	return prefixedData, nil
}

// getDataPrefix returns the encryption prefix
// Format: k8s:enc:<encryption-type>:<encryption-version>:
// For example: k8s:enc:kms:v2:
func getDataPrefix(data []byte) string {
	sData := strings.SplitAfterN(string(data), ":", 5)
	if len(sData) < 5 {
		return ""
	} else {
		return strings.Join(sData[0:4], "")
	}
}
