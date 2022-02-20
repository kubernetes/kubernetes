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

package identity

import (
	"bytes"
	"context"
	"fmt"

	"k8s.io/apiserver/pkg/storage/value"
)

// identityTransformer performs no transformation on provided data, but validates
// that the data is not encrypted data during TransformFromStorage
type identityTransformer struct{}

// NewEncryptCheckTransformer returns an identityTransformer which returns an error
// on attempts to read encrypted data
func NewEncryptCheckTransformer() value.Transformer {
	return identityTransformer{}
}

// TransformFromStorage returns the input bytes if the data is not encrypted
func (identityTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	// identityTransformer has to return an error if the data is encoded using another transformer.
	// JSON data starts with '{'. Protobuf data has a prefix 'k8s[\x00-\xFF]'.
	// Prefix 'k8s:enc:' is reserved for encrypted data on disk.
	if bytes.HasPrefix(data, []byte("k8s:enc:")) {
		return []byte{}, false, fmt.Errorf("identity transformer tried to read encrypted data")
	}
	return data, false, nil
}

// TransformToStorage implements the Transformer interface for identityTransformer
func (identityTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	return data, nil
}
