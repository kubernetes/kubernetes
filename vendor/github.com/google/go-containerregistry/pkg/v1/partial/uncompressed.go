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
	"bytes"
	"io"
	"sync"

	"github.com/google/go-containerregistry/internal/gzip"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// UncompressedLayer represents the bare minimum interface a natively
// uncompressed layer must implement for us to produce a v1.Layer
type UncompressedLayer interface {
	// DiffID returns the Hash of the uncompressed layer.
	DiffID() (v1.Hash, error)

	// Uncompressed returns an io.ReadCloser for the uncompressed layer contents.
	Uncompressed() (io.ReadCloser, error)

	// Returns the mediaType for the compressed Layer
	MediaType() (types.MediaType, error)
}

// uncompressedLayerExtender implements v1.Image using the uncompressed base properties.
type uncompressedLayerExtender struct {
	UncompressedLayer
	// Memoize size/hash so that the methods aren't twice as
	// expensive as doing this manually.
	hash          v1.Hash
	size          int64
	hashSizeError error
	once          sync.Once
}

// Compressed implements v1.Layer
func (ule *uncompressedLayerExtender) Compressed() (io.ReadCloser, error) {
	u, err := ule.Uncompressed()
	if err != nil {
		return nil, err
	}
	return gzip.ReadCloser(u), nil
}

// Digest implements v1.Layer
func (ule *uncompressedLayerExtender) Digest() (v1.Hash, error) {
	ule.calcSizeHash()
	return ule.hash, ule.hashSizeError
}

// Size implements v1.Layer
func (ule *uncompressedLayerExtender) Size() (int64, error) {
	ule.calcSizeHash()
	return ule.size, ule.hashSizeError
}

func (ule *uncompressedLayerExtender) calcSizeHash() {
	ule.once.Do(func() {
		var r io.ReadCloser
		r, ule.hashSizeError = ule.Compressed()
		if ule.hashSizeError != nil {
			return
		}
		defer r.Close()
		ule.hash, ule.size, ule.hashSizeError = v1.SHA256(r)
	})
}

// UncompressedToLayer fills in the missing methods from an UncompressedLayer so that it implements v1.Layer
func UncompressedToLayer(ul UncompressedLayer) (v1.Layer, error) {
	return &uncompressedLayerExtender{UncompressedLayer: ul}, nil
}

// UncompressedImageCore represents the bare minimum interface a natively
// uncompressed image must implement for us to produce a v1.Image
type UncompressedImageCore interface {
	ImageCore

	// LayerByDiffID is a variation on the v1.Image method, which returns
	// an UncompressedLayer instead.
	LayerByDiffID(v1.Hash) (UncompressedLayer, error)
}

// UncompressedToImage fills in the missing methods from an UncompressedImageCore so that it implements v1.Image.
func UncompressedToImage(uic UncompressedImageCore) (v1.Image, error) {
	return &uncompressedImageExtender{
		UncompressedImageCore: uic,
	}, nil
}

// uncompressedImageExtender implements v1.Image by extending UncompressedImageCore with the
// appropriate methods computed from the minimal core.
type uncompressedImageExtender struct {
	UncompressedImageCore

	lock     sync.Mutex
	manifest *v1.Manifest
}

// Assert that our extender type completes the v1.Image interface
var _ v1.Image = (*uncompressedImageExtender)(nil)

// Digest implements v1.Image
func (i *uncompressedImageExtender) Digest() (v1.Hash, error) {
	return Digest(i)
}

// Manifest implements v1.Image
func (i *uncompressedImageExtender) Manifest() (*v1.Manifest, error) {
	i.lock.Lock()
	defer i.lock.Unlock()
	if i.manifest != nil {
		return i.manifest, nil
	}

	b, err := i.RawConfigFile()
	if err != nil {
		return nil, err
	}

	cfgHash, cfgSize, err := v1.SHA256(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	m := &v1.Manifest{
		SchemaVersion: 2,
		MediaType:     types.DockerManifestSchema2,
		Config: v1.Descriptor{
			MediaType: types.DockerConfigJSON,
			Size:      cfgSize,
			Digest:    cfgHash,
		},
	}

	ls, err := i.Layers()
	if err != nil {
		return nil, err
	}

	m.Layers = make([]v1.Descriptor, len(ls))
	for i, l := range ls {
		desc, err := Descriptor(l)
		if err != nil {
			return nil, err
		}

		m.Layers[i] = *desc
	}

	i.manifest = m
	return i.manifest, nil
}

// RawManifest implements v1.Image
func (i *uncompressedImageExtender) RawManifest() ([]byte, error) {
	return RawManifest(i)
}

// Size implements v1.Image
func (i *uncompressedImageExtender) Size() (int64, error) {
	return Size(i)
}

// ConfigName implements v1.Image
func (i *uncompressedImageExtender) ConfigName() (v1.Hash, error) {
	return ConfigName(i)
}

// ConfigFile implements v1.Image
func (i *uncompressedImageExtender) ConfigFile() (*v1.ConfigFile, error) {
	return ConfigFile(i)
}

// Layers implements v1.Image
func (i *uncompressedImageExtender) Layers() ([]v1.Layer, error) {
	diffIDs, err := DiffIDs(i)
	if err != nil {
		return nil, err
	}
	ls := make([]v1.Layer, 0, len(diffIDs))
	for _, h := range diffIDs {
		l, err := i.LayerByDiffID(h)
		if err != nil {
			return nil, err
		}
		ls = append(ls, l)
	}
	return ls, nil
}

// LayerByDiffID implements v1.Image
func (i *uncompressedImageExtender) LayerByDiffID(diffID v1.Hash) (v1.Layer, error) {
	ul, err := i.UncompressedImageCore.LayerByDiffID(diffID)
	if err != nil {
		return nil, err
	}
	return UncompressedToLayer(ul)
}

// LayerByDigest implements v1.Image
func (i *uncompressedImageExtender) LayerByDigest(h v1.Hash) (v1.Layer, error) {
	diffID, err := BlobToDiffID(i, h)
	if err != nil {
		return nil, err
	}
	return i.LayerByDiffID(diffID)
}
