/*
Copyright 2023 The Kubernetes Authors.

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

package encryption

import "context"

// Store is a simple interface to store and retrieve Transformer. It is expected
// to be thread-safe.
type Store interface {
	// Add adds a transformer to the store with its encrypted key as key.
	Add([]byte, Transformer)
	// Get returns a transformer from the store by its encrypted key as key.
	Get([]byte) (Transformer, bool)
}

// CreateTransformer enables the creation of a Transformer based on a key.
type CreateTransformer interface {
	// Transformer creates a transformer with a given key.
	Transformer(context.Context, []byte) (Transformer, error)
	// Key creates a key that should match the expectations of Transformer().
	Key() ([]byte, error)
}

/*
Copied from:
	- "k8s.io/apiserver/pkg/storage/value"
	- "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
*/

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

// Context is additional information that a storage transformation may need to verify the data at rest.
type Context interface {
	// AuthenticatedData should return an array of bytes that describes the current value. If the value changes,
	// the transformer may report the value as unreadable or tampered. This may be nil if no such description exists
	// or is needed. For additional verification, set this to data that strongly identifies the value, such as
	// the key and creation version of the stored data.
	AuthenticatedData() []byte
}

// DefaultContext is a simple implementation of Context for a slice of bytes.
type DefaultContext []byte

// AuthenticatedData returns itself.
func (c DefaultContext) AuthenticatedData() []byte { return c }
