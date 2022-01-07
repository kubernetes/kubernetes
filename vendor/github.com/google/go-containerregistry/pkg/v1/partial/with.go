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
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"

	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// WithRawConfigFile defines the subset of v1.Image used by these helper methods
type WithRawConfigFile interface {
	// RawConfigFile returns the serialized bytes of this image's config file.
	RawConfigFile() ([]byte, error)
}

// ConfigFile is a helper for implementing v1.Image
func ConfigFile(i WithRawConfigFile) (*v1.ConfigFile, error) {
	b, err := i.RawConfigFile()
	if err != nil {
		return nil, err
	}
	return v1.ParseConfigFile(bytes.NewReader(b))
}

// ConfigName is a helper for implementing v1.Image
func ConfigName(i WithRawConfigFile) (v1.Hash, error) {
	b, err := i.RawConfigFile()
	if err != nil {
		return v1.Hash{}, err
	}
	h, _, err := v1.SHA256(bytes.NewReader(b))
	return h, err
}

type configLayer struct {
	hash    v1.Hash
	content []byte
}

// Digest implements v1.Layer
func (cl *configLayer) Digest() (v1.Hash, error) {
	return cl.hash, nil
}

// DiffID implements v1.Layer
func (cl *configLayer) DiffID() (v1.Hash, error) {
	return cl.hash, nil
}

// Uncompressed implements v1.Layer
func (cl *configLayer) Uncompressed() (io.ReadCloser, error) {
	return ioutil.NopCloser(bytes.NewBuffer(cl.content)), nil
}

// Compressed implements v1.Layer
func (cl *configLayer) Compressed() (io.ReadCloser, error) {
	return ioutil.NopCloser(bytes.NewBuffer(cl.content)), nil
}

// Size implements v1.Layer
func (cl *configLayer) Size() (int64, error) {
	return int64(len(cl.content)), nil
}

func (cl *configLayer) MediaType() (types.MediaType, error) {
	// Defaulting this to OCIConfigJSON as it should remain
	// backwards compatible with DockerConfigJSON
	return types.OCIConfigJSON, nil
}

var _ v1.Layer = (*configLayer)(nil)

// ConfigLayer implements v1.Layer from the raw config bytes.
// This is so that clients (e.g. remote) can access the config as a blob.
func ConfigLayer(i WithRawConfigFile) (v1.Layer, error) {
	h, err := ConfigName(i)
	if err != nil {
		return nil, err
	}
	rcfg, err := i.RawConfigFile()
	if err != nil {
		return nil, err
	}
	return &configLayer{
		hash:    h,
		content: rcfg,
	}, nil
}

// WithConfigFile defines the subset of v1.Image used by these helper methods
type WithConfigFile interface {
	// ConfigFile returns this image's config file.
	ConfigFile() (*v1.ConfigFile, error)
}

// DiffIDs is a helper for implementing v1.Image
func DiffIDs(i WithConfigFile) ([]v1.Hash, error) {
	cfg, err := i.ConfigFile()
	if err != nil {
		return nil, err
	}
	return cfg.RootFS.DiffIDs, nil
}

// RawConfigFile is a helper for implementing v1.Image
func RawConfigFile(i WithConfigFile) ([]byte, error) {
	cfg, err := i.ConfigFile()
	if err != nil {
		return nil, err
	}
	return json.Marshal(cfg)
}

// WithRawManifest defines the subset of v1.Image used by these helper methods
type WithRawManifest interface {
	// RawManifest returns the serialized bytes of this image's config file.
	RawManifest() ([]byte, error)
}

// Digest is a helper for implementing v1.Image
func Digest(i WithRawManifest) (v1.Hash, error) {
	mb, err := i.RawManifest()
	if err != nil {
		return v1.Hash{}, err
	}
	digest, _, err := v1.SHA256(bytes.NewReader(mb))
	return digest, err
}

// Manifest is a helper for implementing v1.Image
func Manifest(i WithRawManifest) (*v1.Manifest, error) {
	b, err := i.RawManifest()
	if err != nil {
		return nil, err
	}
	return v1.ParseManifest(bytes.NewReader(b))
}

// WithManifest defines the subset of v1.Image used by these helper methods
type WithManifest interface {
	// Manifest returns this image's Manifest object.
	Manifest() (*v1.Manifest, error)
}

// RawManifest is a helper for implementing v1.Image
func RawManifest(i WithManifest) ([]byte, error) {
	m, err := i.Manifest()
	if err != nil {
		return nil, err
	}
	return json.Marshal(m)
}

// Size is a helper for implementing v1.Image
func Size(i WithRawManifest) (int64, error) {
	b, err := i.RawManifest()
	if err != nil {
		return -1, err
	}
	return int64(len(b)), nil
}

// FSLayers is a helper for implementing v1.Image
func FSLayers(i WithManifest) ([]v1.Hash, error) {
	m, err := i.Manifest()
	if err != nil {
		return nil, err
	}
	fsl := make([]v1.Hash, len(m.Layers))
	for i, l := range m.Layers {
		fsl[i] = l.Digest
	}
	return fsl, nil
}

// BlobSize is a helper for implementing v1.Image
func BlobSize(i WithManifest, h v1.Hash) (int64, error) {
	d, err := BlobDescriptor(i, h)
	if err != nil {
		return -1, err
	}
	return d.Size, nil
}

// BlobDescriptor is a helper for implementing v1.Image
func BlobDescriptor(i WithManifest, h v1.Hash) (*v1.Descriptor, error) {
	m, err := i.Manifest()
	if err != nil {
		return nil, err
	}

	if m.Config.Digest == h {
		return &m.Config, nil
	}

	for _, l := range m.Layers {
		if l.Digest == h {
			return &l, nil
		}
	}
	return nil, fmt.Errorf("blob %v not found", h)
}

// WithManifestAndConfigFile defines the subset of v1.Image used by these helper methods
type WithManifestAndConfigFile interface {
	WithConfigFile

	// Manifest returns this image's Manifest object.
	Manifest() (*v1.Manifest, error)
}

// BlobToDiffID is a helper for mapping between compressed
// and uncompressed blob hashes.
func BlobToDiffID(i WithManifestAndConfigFile, h v1.Hash) (v1.Hash, error) {
	blobs, err := FSLayers(i)
	if err != nil {
		return v1.Hash{}, err
	}
	diffIDs, err := DiffIDs(i)
	if err != nil {
		return v1.Hash{}, err
	}
	if len(blobs) != len(diffIDs) {
		return v1.Hash{}, fmt.Errorf("mismatched fs layers (%d) and diff ids (%d)", len(blobs), len(diffIDs))
	}
	for i, blob := range blobs {
		if blob == h {
			return diffIDs[i], nil
		}
	}
	return v1.Hash{}, fmt.Errorf("unknown blob %v", h)
}

// DiffIDToBlob is a helper for mapping between uncompressed
// and compressed blob hashes.
func DiffIDToBlob(wm WithManifestAndConfigFile, h v1.Hash) (v1.Hash, error) {
	blobs, err := FSLayers(wm)
	if err != nil {
		return v1.Hash{}, err
	}
	diffIDs, err := DiffIDs(wm)
	if err != nil {
		return v1.Hash{}, err
	}
	if len(blobs) != len(diffIDs) {
		return v1.Hash{}, fmt.Errorf("mismatched fs layers (%d) and diff ids (%d)", len(blobs), len(diffIDs))
	}
	for i, diffID := range diffIDs {
		if diffID == h {
			return blobs[i], nil
		}
	}
	return v1.Hash{}, fmt.Errorf("unknown diffID %v", h)
}

// WithDiffID defines the subset of v1.Layer for exposing the DiffID method.
type WithDiffID interface {
	DiffID() (v1.Hash, error)
}

// withDescriptor allows partial layer implementations to provide a layer
// descriptor to the partial image manifest builder. This allows partial
// uncompressed layers to provide foreign layer metadata like URLs to the
// uncompressed image manifest.
type withDescriptor interface {
	Descriptor() (*v1.Descriptor, error)
}

// Describable represents something for which we can produce a v1.Descriptor.
type Describable interface {
	Digest() (v1.Hash, error)
	MediaType() (types.MediaType, error)
	Size() (int64, error)
}

// Descriptor returns a v1.Descriptor given a Describable. It also encodes
// some logic for unwrapping things that have been wrapped by
// CompressedToLayer, UncompressedToLayer, CompressedToImage, or
// UncompressedToImage.
func Descriptor(d Describable) (*v1.Descriptor, error) {
	// If Describable implements Descriptor itself, return that.
	if wd, ok := unwrap(d).(withDescriptor); ok {
		return wd.Descriptor()
	}

	// If all else fails, compute the descriptor from the individual methods.
	var (
		desc v1.Descriptor
		err  error
	)

	if desc.Size, err = d.Size(); err != nil {
		return nil, err
	}
	if desc.Digest, err = d.Digest(); err != nil {
		return nil, err
	}
	if desc.MediaType, err = d.MediaType(); err != nil {
		return nil, err
	}

	return &desc, nil
}

type withUncompressedSize interface {
	UncompressedSize() (int64, error)
}

// UncompressedSize returns the size of the Uncompressed layer. If the
// underlying implementation doesn't implement UncompressedSize directly,
// this will compute the uncompressedSize by reading everything returned
// by Compressed(). This is potentially expensive and may consume the contents
// for streaming layers.
func UncompressedSize(l v1.Layer) (int64, error) {
	// If the layer implements UncompressedSize itself, return that.
	if wus, ok := unwrap(l).(withUncompressedSize); ok {
		return wus.UncompressedSize()
	}

	// The layer doesn't implement UncompressedSize, we need to compute it.
	rc, err := l.Uncompressed()
	if err != nil {
		return -1, err
	}
	defer rc.Close()

	return io.Copy(ioutil.Discard, rc)
}

type withExists interface {
	Exists() (bool, error)
}

// Exists checks to see if a layer exists. This is a hack to work around the
// mistakes of the partial package. Don't use this.
func Exists(l v1.Layer) (bool, error) {
	// If the layer implements Exists itself, return that.
	if we, ok := unwrap(l).(withExists); ok {
		return we.Exists()
	}

	// The layer doesn't implement Exists, so we hope that calling Compressed()
	// is enough to trigger an error if the layer does not exist.
	rc, err := l.Compressed()
	if err != nil {
		return false, err
	}
	defer rc.Close()

	// We may want to try actually reading a single byte, but if we need to do
	// that, we should just fix this hack.
	return true, nil
}

// Recursively unwrap our wrappers so that we can check for the original implementation.
// We might want to expose this?
func unwrap(i interface{}) interface{} {
	if ule, ok := i.(*uncompressedLayerExtender); ok {
		return unwrap(ule.UncompressedLayer)
	}
	if cle, ok := i.(*compressedLayerExtender); ok {
		return unwrap(cle.CompressedLayer)
	}
	if uie, ok := i.(*uncompressedImageExtender); ok {
		return unwrap(uie.UncompressedImageCore)
	}
	if cie, ok := i.(*compressedImageExtender); ok {
		return unwrap(cie.CompressedImageCore)
	}
	return i
}
