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

package partial

import (
	"io"

	"github.com/google/go-containerregistry/internal/gzip"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// CompressedLayer represents the bare minimum interface a natively
// compressed layer must implement for us to produce a v1.Layer
type CompressedLayer interface {
	// Digest returns the Hash of the compressed layer.
	Digest() (v1.Hash, error)

	// Compressed returns an io.ReadCloser for the compressed layer contents.
	Compressed() (io.ReadCloser, error)

	// Size returns the compressed size of the Layer.
	Size() (int64, error)

	// Returns the mediaType for the compressed Layer
	MediaType() (types.MediaType, error)
}

// compressedLayerExtender implements v1.Image using the compressed base properties.
type compressedLayerExtender struct {
	CompressedLayer
}

// Uncompressed implements v1.Layer
func (cle *compressedLayerExtender) Uncompressed() (io.ReadCloser, error) {
	r, err := cle.Compressed()
	if err != nil {
		return nil, err
	}
	return gzip.UnzipReadCloser(r)
}

// DiffID implements v1.Layer
func (cle *compressedLayerExtender) DiffID() (v1.Hash, error) {
	// If our nested CompressedLayer implements DiffID,
	// then delegate to it instead.
	if wdi, ok := cle.CompressedLayer.(WithDiffID); ok {
		return wdi.DiffID()
	}
	r, err := cle.Uncompressed()
	if err != nil {
		return v1.Hash{}, err
	}
	defer r.Close()
	h, _, err := v1.SHA256(r)
	return h, err
}

// CompressedToLayer fills in the missing methods from a CompressedLayer so that it implements v1.Layer
func CompressedToLayer(ul CompressedLayer) (v1.Layer, error) {
	return &compressedLayerExtender{ul}, nil
}

// CompressedImageCore represents the base minimum interface a natively
// compressed image must implement for us to produce a v1.Image.
type CompressedImageCore interface {
	ImageCore

	// RawManifest returns the serialized bytes of the manifest.
	RawManifest() ([]byte, error)

	// LayerByDigest is a variation on the v1.Image method, which returns
	// a CompressedLayer instead.
	LayerByDigest(v1.Hash) (CompressedLayer, error)
}

// compressedImageExtender implements v1.Image by extending CompressedImageCore with the
// appropriate methods computed from the minimal core.
type compressedImageExtender struct {
	CompressedImageCore
}

// Assert that our extender type completes the v1.Image interface
var _ v1.Image = (*compressedImageExtender)(nil)

// Digest implements v1.Image
func (i *compressedImageExtender) Digest() (v1.Hash, error) {
	return Digest(i)
}

// ConfigName implements v1.Image
func (i *compressedImageExtender) ConfigName() (v1.Hash, error) {
	return ConfigName(i)
}

// Layers implements v1.Image
func (i *compressedImageExtender) Layers() ([]v1.Layer, error) {
	hs, err := FSLayers(i)
	if err != nil {
		return nil, err
	}
	ls := make([]v1.Layer, 0, len(hs))
	for _, h := range hs {
		l, err := i.LayerByDigest(h)
		if err != nil {
			return nil, err
		}
		ls = append(ls, l)
	}
	return ls, nil
}

// LayerByDigest implements v1.Image
func (i *compressedImageExtender) LayerByDigest(h v1.Hash) (v1.Layer, error) {
	cl, err := i.CompressedImageCore.LayerByDigest(h)
	if err != nil {
		return nil, err
	}
	return CompressedToLayer(cl)
}

// LayerByDiffID implements v1.Image
func (i *compressedImageExtender) LayerByDiffID(h v1.Hash) (v1.Layer, error) {
	h, err := DiffIDToBlob(i, h)
	if err != nil {
		return nil, err
	}
	return i.LayerByDigest(h)
}

// ConfigFile implements v1.Image
func (i *compressedImageExtender) ConfigFile() (*v1.ConfigFile, error) {
	return ConfigFile(i)
}

// Manifest implements v1.Image
func (i *compressedImageExtender) Manifest() (*v1.Manifest, error) {
	return Manifest(i)
}

// Size implements v1.Image
func (i *compressedImageExtender) Size() (int64, error) {
	return Size(i)
}

// CompressedToImage fills in the missing methods from a CompressedImageCore so that it implements v1.Image
func CompressedToImage(cic CompressedImageCore) (v1.Image, error) {
	return &compressedImageExtender{
		CompressedImageCore: cic,
	}, nil
}
