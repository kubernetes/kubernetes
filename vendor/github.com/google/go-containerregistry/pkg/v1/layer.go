// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v1

import (
	"io"

	"github.com/google/go-containerregistry/pkg/v1/types"
)

// Layer is an interface for accessing the properties of a particular layer of a v1.Image
type Layer interface {
	// Digest returns the Hash of the compressed layer.
	Digest() (Hash, error)

	// DiffID returns the Hash of the uncompressed layer.
	DiffID() (Hash, error)

	// Compressed returns an io.ReadCloser for the compressed layer contents.
	Compressed() (io.ReadCloser, error)

	// Uncompressed returns an io.ReadCloser for the uncompressed layer contents.
	Uncompressed() (io.ReadCloser, error)

	// Size returns the compressed size of the Layer.
	Size() (int64, error)

	// MediaType returns the media type of the Layer.
	MediaType() (types.MediaType, error)
}
