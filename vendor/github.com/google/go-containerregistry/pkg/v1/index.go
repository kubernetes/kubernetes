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
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// ImageIndex defines the interface for interacting with an OCI image index.
type ImageIndex interface {
	// MediaType of this image's manifest.
	MediaType() (types.MediaType, error)

	// Digest returns the sha256 of this index's manifest.
	Digest() (Hash, error)

	// Size returns the size of the manifest.
	Size() (int64, error)

	// IndexManifest returns this image index's manifest object.
	IndexManifest() (*IndexManifest, error)

	// RawManifest returns the serialized bytes of IndexManifest().
	RawManifest() ([]byte, error)

	// Image returns a v1.Image that this ImageIndex references.
	Image(Hash) (Image, error)

	// ImageIndex returns a v1.ImageIndex that this ImageIndex references.
	ImageIndex(Hash) (ImageIndex, error)
}
