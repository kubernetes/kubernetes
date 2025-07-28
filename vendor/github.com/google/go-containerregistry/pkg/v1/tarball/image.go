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

package tarball

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sync"

	comp "github.com/google/go-containerregistry/internal/compression"
	"github.com/google/go-containerregistry/pkg/compression"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

type image struct {
	opener        Opener
	manifest      *Manifest
	config        []byte
	imgDescriptor *Descriptor

	tag *name.Tag
}

type uncompressedImage struct {
	*image
}

type compressedImage struct {
	*image
	manifestLock sync.Mutex // Protects manifest
	manifest     *v1.Manifest
}

var _ partial.UncompressedImageCore = (*uncompressedImage)(nil)
var _ partial.CompressedImageCore = (*compressedImage)(nil)

// Opener is a thunk for opening a tar file.
type Opener func() (io.ReadCloser, error)

func pathOpener(path string) Opener {
	return func() (io.ReadCloser, error) {
		return os.Open(path)
	}
}

// ImageFromPath returns a v1.Image from a tarball located on path.
func ImageFromPath(path string, tag *name.Tag) (v1.Image, error) {
	return Image(pathOpener(path), tag)
}

// LoadManifest load manifest
func LoadManifest(opener Opener) (Manifest, error) {
	m, err := extractFileFromTar(opener, "manifest.json")
	if err != nil {
		return nil, err
	}
	defer m.Close()

	var manifest Manifest

	if err := json.NewDecoder(m).Decode(&manifest); err != nil {
		return nil, err
	}
	return manifest, nil
}

// Image exposes an image from the tarball at the provided path.
func Image(opener Opener, tag *name.Tag) (v1.Image, error) {
	img := &image{
		opener: opener,
		tag:    tag,
	}
	if err := img.loadTarDescriptorAndConfig(); err != nil {
		return nil, err
	}

	// Peek at the first layer and see if it's compressed.
	if len(img.imgDescriptor.Layers) > 0 {
		compressed, err := img.areLayersCompressed()
		if err != nil {
			return nil, err
		}
		if compressed {
			c := compressedImage{
				image: img,
			}
			return partial.CompressedToImage(&c)
		}
	}

	uc := uncompressedImage{
		image: img,
	}
	return partial.UncompressedToImage(&uc)
}

func (i *image) MediaType() (types.MediaType, error) {
	return types.DockerManifestSchema2, nil
}

// Descriptor stores the manifest data for a single image inside a `docker save` tarball.
type Descriptor struct {
	Config   string
	RepoTags []string
	Layers   []string

	// Tracks foreign layer info. Key is DiffID.
	LayerSources map[v1.Hash]v1.Descriptor `json:",omitempty"`
}

// Manifest represents the manifests of all images as the `manifest.json` file in a `docker save` tarball.
type Manifest []Descriptor

func (m Manifest) findDescriptor(tag *name.Tag) (*Descriptor, error) {
	if tag == nil {
		if len(m) != 1 {
			return nil, errors.New("tarball must contain only a single image to be used with tarball.Image")
		}
		return &(m)[0], nil
	}
	for _, img := range m {
		for _, tagStr := range img.RepoTags {
			repoTag, err := name.NewTag(tagStr)
			if err != nil {
				return nil, err
			}

			// Compare the resolved names, since there are several ways to specify the same tag.
			if repoTag.Name() == tag.Name() {
				return &img, nil
			}
		}
	}
	return nil, fmt.Errorf("tag %s not found in tarball", tag)
}

func (i *image) areLayersCompressed() (bool, error) {
	if len(i.imgDescriptor.Layers) == 0 {
		return false, errors.New("0 layers found in image")
	}
	layer := i.imgDescriptor.Layers[0]
	blob, err := extractFileFromTar(i.opener, layer)
	if err != nil {
		return false, err
	}
	defer blob.Close()

	cp, _, err := comp.PeekCompression(blob)
	if err != nil {
		return false, err
	}

	return cp != compression.None, nil
}

func (i *image) loadTarDescriptorAndConfig() error {
	m, err := extractFileFromTar(i.opener, "manifest.json")
	if err != nil {
		return err
	}
	defer m.Close()

	if err := json.NewDecoder(m).Decode(&i.manifest); err != nil {
		return err
	}

	if i.manifest == nil {
		return errors.New("no valid manifest.json in tarball")
	}

	i.imgDescriptor, err = i.manifest.findDescriptor(i.tag)
	if err != nil {
		return err
	}

	cfg, err := extractFileFromTar(i.opener, i.imgDescriptor.Config)
	if err != nil {
		return err
	}
	defer cfg.Close()

	i.config, err = io.ReadAll(cfg)
	if err != nil {
		return err
	}
	return nil
}

func (i *image) RawConfigFile() ([]byte, error) {
	return i.config, nil
}

// tarFile represents a single file inside a tar. Closing it closes the tar itself.
type tarFile struct {
	io.Reader
	io.Closer
}

func extractFileFromTar(opener Opener, filePath string) (io.ReadCloser, error) {
	f, err := opener()
	if err != nil {
		return nil, err
	}
	needClose := true
	defer func() {
		if needClose {
			f.Close()
		}
	}()

	tf := tar.NewReader(f)
	for {
		hdr, err := tf.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		if hdr.Name == filePath {
			if hdr.Typeflag == tar.TypeSymlink || hdr.Typeflag == tar.TypeLink {
				currentDir := filepath.Dir(filePath)
				return extractFileFromTar(opener, path.Join(currentDir, path.Clean(hdr.Linkname)))
			}
			needClose = false
			return tarFile{
				Reader: tf,
				Closer: f,
			}, nil
		}
	}
	return nil, fmt.Errorf("file %s not found in tar", filePath)
}

// uncompressedLayerFromTarball implements partial.UncompressedLayer
type uncompressedLayerFromTarball struct {
	diffID    v1.Hash
	mediaType types.MediaType
	opener    Opener
	filePath  string
}

// foreignUncompressedLayer implements partial.UncompressedLayer but returns
// a custom descriptor. This allows the foreign layer URLs to be included in
// the generated image manifest for uncompressed layers.
type foreignUncompressedLayer struct {
	uncompressedLayerFromTarball
	desc v1.Descriptor
}

func (fl *foreignUncompressedLayer) Descriptor() (*v1.Descriptor, error) {
	return &fl.desc, nil
}

// DiffID implements partial.UncompressedLayer
func (ulft *uncompressedLayerFromTarball) DiffID() (v1.Hash, error) {
	return ulft.diffID, nil
}

// Uncompressed implements partial.UncompressedLayer
func (ulft *uncompressedLayerFromTarball) Uncompressed() (io.ReadCloser, error) {
	return extractFileFromTar(ulft.opener, ulft.filePath)
}

func (ulft *uncompressedLayerFromTarball) MediaType() (types.MediaType, error) {
	return ulft.mediaType, nil
}

func (i *uncompressedImage) LayerByDiffID(h v1.Hash) (partial.UncompressedLayer, error) {
	cfg, err := partial.ConfigFile(i)
	if err != nil {
		return nil, err
	}
	for idx, diffID := range cfg.RootFS.DiffIDs {
		if diffID == h {
			// Technically the media type should be 'application/tar' but given that our
			// v1.Layer doesn't force consumers to care about whether the layer is compressed
			// we should be fine returning the DockerLayer media type
			mt := types.DockerLayer
			bd, ok := i.imgDescriptor.LayerSources[h]
			if ok {
				// This is janky, but we don't want to implement Descriptor for
				// uncompressed layers because it breaks a bunch of assumptions in partial.
				// See https://github.com/google/go-containerregistry/issues/1870
				docker25workaround := bd.MediaType == types.DockerUncompressedLayer || bd.MediaType == types.OCIUncompressedLayer

				if !docker25workaround {
					// Overwrite the mediaType for foreign layers.
					return &foreignUncompressedLayer{
						uncompressedLayerFromTarball: uncompressedLayerFromTarball{
							diffID:    diffID,
							mediaType: bd.MediaType,
							opener:    i.opener,
							filePath:  i.imgDescriptor.Layers[idx],
						},
						desc: bd,
					}, nil
				}

				// Intentional fall through.
			}

			return &uncompressedLayerFromTarball{
				diffID:    diffID,
				mediaType: mt,
				opener:    i.opener,
				filePath:  i.imgDescriptor.Layers[idx],
			}, nil
		}
	}
	return nil, fmt.Errorf("diff id %q not found", h)
}

func (c *compressedImage) Manifest() (*v1.Manifest, error) {
	c.manifestLock.Lock()
	defer c.manifestLock.Unlock()
	if c.manifest != nil {
		return c.manifest, nil
	}

	b, err := c.RawConfigFile()
	if err != nil {
		return nil, err
	}

	cfgHash, cfgSize, err := v1.SHA256(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	c.manifest = &v1.Manifest{
		SchemaVersion: 2,
		MediaType:     types.DockerManifestSchema2,
		Config: v1.Descriptor{
			MediaType: types.DockerConfigJSON,
			Size:      cfgSize,
			Digest:    cfgHash,
		},
	}

	for i, p := range c.imgDescriptor.Layers {
		cfg, err := partial.ConfigFile(c)
		if err != nil {
			return nil, err
		}
		diffid := cfg.RootFS.DiffIDs[i]
		if d, ok := c.imgDescriptor.LayerSources[diffid]; ok {
			// If it's a foreign layer, just append the descriptor so we can avoid
			// reading the entire file.
			c.manifest.Layers = append(c.manifest.Layers, d)
		} else {
			l, err := extractFileFromTar(c.opener, p)
			if err != nil {
				return nil, err
			}
			defer l.Close()
			sha, size, err := v1.SHA256(l)
			if err != nil {
				return nil, err
			}
			c.manifest.Layers = append(c.manifest.Layers, v1.Descriptor{
				MediaType: types.DockerLayer,
				Size:      size,
				Digest:    sha,
			})
		}
	}
	return c.manifest, nil
}

func (c *compressedImage) RawManifest() ([]byte, error) {
	return partial.RawManifest(c)
}

// compressedLayerFromTarball implements partial.CompressedLayer
type compressedLayerFromTarball struct {
	desc     v1.Descriptor
	opener   Opener
	filePath string
}

// Digest implements partial.CompressedLayer
func (clft *compressedLayerFromTarball) Digest() (v1.Hash, error) {
	return clft.desc.Digest, nil
}

// Compressed implements partial.CompressedLayer
func (clft *compressedLayerFromTarball) Compressed() (io.ReadCloser, error) {
	return extractFileFromTar(clft.opener, clft.filePath)
}

// MediaType implements partial.CompressedLayer
func (clft *compressedLayerFromTarball) MediaType() (types.MediaType, error) {
	return clft.desc.MediaType, nil
}

// Size implements partial.CompressedLayer
func (clft *compressedLayerFromTarball) Size() (int64, error) {
	return clft.desc.Size, nil
}

func (c *compressedImage) LayerByDigest(h v1.Hash) (partial.CompressedLayer, error) {
	m, err := c.Manifest()
	if err != nil {
		return nil, err
	}
	for i, l := range m.Layers {
		if l.Digest == h {
			fp := c.imgDescriptor.Layers[i]
			return &compressedLayerFromTarball{
				desc:     l,
				opener:   c.opener,
				filePath: fp,
			}, nil
		}
	}
	return nil, fmt.Errorf("blob %v not found", h)
}
