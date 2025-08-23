// Copyright 2019 Google LLC All Rights Reserved.
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

package mutate

import (
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/google/go-containerregistry/pkg/logs"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/match"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/stream"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

func computeDescriptor(ia IndexAddendum) (*v1.Descriptor, error) {
	desc, err := partial.Descriptor(ia.Add)
	if err != nil {
		return nil, err
	}

	// The IndexAddendum allows overriding Descriptor values.
	if ia.Size != 0 {
		desc.Size = ia.Size
	}
	if string(ia.MediaType) != "" {
		desc.MediaType = ia.MediaType
	}
	if ia.Digest != (v1.Hash{}) {
		desc.Digest = ia.Digest
	}
	if ia.Platform != nil {
		desc.Platform = ia.Platform
	}
	if len(ia.URLs) != 0 {
		desc.URLs = ia.URLs
	}
	if len(ia.Annotations) != 0 {
		desc.Annotations = ia.Annotations
	}
	if ia.Data != nil {
		desc.Data = ia.Data
	}

	return desc, nil
}

type index struct {
	base v1.ImageIndex
	adds []IndexAddendum
	// remove is removed before adds
	remove match.Matcher

	computed    bool
	manifest    *v1.IndexManifest
	annotations map[string]string
	mediaType   *types.MediaType
	imageMap    map[v1.Hash]v1.Image
	indexMap    map[v1.Hash]v1.ImageIndex
	layerMap    map[v1.Hash]v1.Layer
	subject     *v1.Descriptor

	sync.Mutex
}

var _ v1.ImageIndex = (*index)(nil)

func (i *index) MediaType() (types.MediaType, error) {
	if i.mediaType != nil {
		return *i.mediaType, nil
	}
	return i.base.MediaType()
}

func (i *index) Size() (int64, error) { return partial.Size(i) }

func (i *index) compute() error {
	i.Lock()
	defer i.Unlock()

	// Don't re-compute if already computed.
	if i.computed {
		return nil
	}

	i.imageMap = make(map[v1.Hash]v1.Image)
	i.indexMap = make(map[v1.Hash]v1.ImageIndex)
	i.layerMap = make(map[v1.Hash]v1.Layer)

	m, err := i.base.IndexManifest()
	if err != nil {
		return err
	}
	manifest := m.DeepCopy()
	manifests := manifest.Manifests

	if i.remove != nil {
		var cleanedManifests []v1.Descriptor
		for _, m := range manifests {
			if !i.remove(m) {
				cleanedManifests = append(cleanedManifests, m)
			}
		}
		manifests = cleanedManifests
	}

	for _, add := range i.adds {
		desc, err := computeDescriptor(add)
		if err != nil {
			return err
		}

		manifests = append(manifests, *desc)
		if idx, ok := add.Add.(v1.ImageIndex); ok {
			i.indexMap[desc.Digest] = idx
		} else if img, ok := add.Add.(v1.Image); ok {
			i.imageMap[desc.Digest] = img
		} else if l, ok := add.Add.(v1.Layer); ok {
			i.layerMap[desc.Digest] = l
		} else {
			logs.Warn.Printf("Unexpected index addendum: %T", add.Add)
		}
	}

	manifest.Manifests = manifests

	if i.mediaType != nil {
		manifest.MediaType = *i.mediaType
	}

	if i.annotations != nil {
		if manifest.Annotations == nil {
			manifest.Annotations = map[string]string{}
		}
		for k, v := range i.annotations {
			manifest.Annotations[k] = v
		}
	}
	manifest.Subject = i.subject

	i.manifest = manifest
	i.computed = true
	return nil
}

func (i *index) Image(h v1.Hash) (v1.Image, error) {
	if img, ok := i.imageMap[h]; ok {
		return img, nil
	}
	return i.base.Image(h)
}

func (i *index) ImageIndex(h v1.Hash) (v1.ImageIndex, error) {
	if idx, ok := i.indexMap[h]; ok {
		return idx, nil
	}
	return i.base.ImageIndex(h)
}

type withLayer interface {
	Layer(v1.Hash) (v1.Layer, error)
}

// Workaround for #819.
func (i *index) Layer(h v1.Hash) (v1.Layer, error) {
	if layer, ok := i.layerMap[h]; ok {
		return layer, nil
	}
	if wl, ok := i.base.(withLayer); ok {
		return wl.Layer(h)
	}
	return nil, fmt.Errorf("layer not found: %s", h)
}

// Digest returns the sha256 of this image's manifest.
func (i *index) Digest() (v1.Hash, error) {
	if err := i.compute(); err != nil {
		return v1.Hash{}, err
	}
	return partial.Digest(i)
}

// Manifest returns this image's Manifest object.
func (i *index) IndexManifest() (*v1.IndexManifest, error) {
	if err := i.compute(); err != nil {
		return nil, err
	}
	return i.manifest.DeepCopy(), nil
}

// RawManifest returns the serialized bytes of Manifest()
func (i *index) RawManifest() ([]byte, error) {
	if err := i.compute(); err != nil {
		return nil, err
	}
	return json.Marshal(i.manifest)
}

func (i *index) Manifests() ([]partial.Describable, error) {
	if err := i.compute(); errors.Is(err, stream.ErrNotComputed) {
		// Index contains a streamable layer which has not yet been
		// consumed. Just return the manifests we have in case the caller
		// is going to consume the streamable layers.
		manifests, err := partial.Manifests(i.base)
		if err != nil {
			return nil, err
		}
		for _, add := range i.adds {
			manifests = append(manifests, add.Add)
		}
		return manifests, nil
	} else if err != nil {
		return nil, err
	}

	return partial.ComputeManifests(i)
}
