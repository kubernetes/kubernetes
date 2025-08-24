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
	"bytes"
	"encoding/json"
	"errors"
	"sync"

	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/stream"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

type image struct {
	base v1.Image
	adds []Addendum

	computed        bool
	configFile      *v1.ConfigFile
	manifest        *v1.Manifest
	annotations     map[string]string
	mediaType       *types.MediaType
	configMediaType *types.MediaType
	diffIDMap       map[v1.Hash]v1.Layer
	digestMap       map[v1.Hash]v1.Layer
	subject         *v1.Descriptor

	sync.Mutex
}

var _ v1.Image = (*image)(nil)

func (i *image) MediaType() (types.MediaType, error) {
	if i.mediaType != nil {
		return *i.mediaType, nil
	}
	return i.base.MediaType()
}

func (i *image) compute() error {
	i.Lock()
	defer i.Unlock()

	// Don't re-compute if already computed.
	if i.computed {
		return nil
	}
	var configFile *v1.ConfigFile
	if i.configFile != nil {
		configFile = i.configFile
	} else {
		cf, err := i.base.ConfigFile()
		if err != nil {
			return err
		}
		configFile = cf.DeepCopy()
	}
	diffIDs := configFile.RootFS.DiffIDs
	history := configFile.History

	diffIDMap := make(map[v1.Hash]v1.Layer)
	digestMap := make(map[v1.Hash]v1.Layer)

	for _, add := range i.adds {
		history = append(history, add.History)
		if add.Layer != nil {
			diffID, err := add.Layer.DiffID()
			if err != nil {
				return err
			}
			diffIDs = append(diffIDs, diffID)
			diffIDMap[diffID] = add.Layer
		}
	}

	m, err := i.base.Manifest()
	if err != nil {
		return err
	}
	manifest := m.DeepCopy()
	manifestLayers := manifest.Layers
	for _, add := range i.adds {
		if add.Layer == nil {
			// Empty layers include only history in manifest.
			continue
		}

		desc, err := partial.Descriptor(add.Layer)
		if err != nil {
			return err
		}

		// Fields in the addendum override the original descriptor.
		if len(add.Annotations) != 0 {
			desc.Annotations = add.Annotations
		}
		if len(add.URLs) != 0 {
			desc.URLs = add.URLs
		}

		if add.MediaType != "" {
			desc.MediaType = add.MediaType
		}

		manifestLayers = append(manifestLayers, *desc)
		digestMap[desc.Digest] = add.Layer
	}

	configFile.RootFS.DiffIDs = diffIDs
	configFile.History = history

	manifest.Layers = manifestLayers

	rcfg, err := json.Marshal(configFile)
	if err != nil {
		return err
	}
	d, sz, err := v1.SHA256(bytes.NewBuffer(rcfg))
	if err != nil {
		return err
	}
	manifest.Config.Digest = d
	manifest.Config.Size = sz

	// If Data was set in the base image, we need to update it in the mutated image.
	if m.Config.Data != nil {
		manifest.Config.Data = rcfg
	}

	// If the user wants to mutate the media type of the config
	if i.configMediaType != nil {
		manifest.Config.MediaType = *i.configMediaType
	}

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

	i.configFile = configFile
	i.manifest = manifest
	i.diffIDMap = diffIDMap
	i.digestMap = digestMap
	i.computed = true
	return nil
}

// Layers returns the ordered collection of filesystem layers that comprise this image.
// The order of the list is oldest/base layer first, and most-recent/top layer last.
func (i *image) Layers() ([]v1.Layer, error) {
	if err := i.compute(); errors.Is(err, stream.ErrNotComputed) {
		// Image contains a streamable layer which has not yet been
		// consumed. Just return the layers we have in case the caller
		// is going to consume the layers.
		layers, err := i.base.Layers()
		if err != nil {
			return nil, err
		}
		for _, add := range i.adds {
			layers = append(layers, add.Layer)
		}
		return layers, nil
	} else if err != nil {
		return nil, err
	}

	diffIDs, err := partial.DiffIDs(i)
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

// ConfigName returns the hash of the image's config file.
func (i *image) ConfigName() (v1.Hash, error) {
	if err := i.compute(); err != nil {
		return v1.Hash{}, err
	}
	return partial.ConfigName(i)
}

// ConfigFile returns this image's config file.
func (i *image) ConfigFile() (*v1.ConfigFile, error) {
	if err := i.compute(); err != nil {
		return nil, err
	}
	return i.configFile.DeepCopy(), nil
}

// RawConfigFile returns the serialized bytes of ConfigFile()
func (i *image) RawConfigFile() ([]byte, error) {
	if err := i.compute(); err != nil {
		return nil, err
	}
	return json.Marshal(i.configFile)
}

// Digest returns the sha256 of this image's manifest.
func (i *image) Digest() (v1.Hash, error) {
	if err := i.compute(); err != nil {
		return v1.Hash{}, err
	}
	return partial.Digest(i)
}

// Size implements v1.Image.
func (i *image) Size() (int64, error) {
	if err := i.compute(); err != nil {
		return -1, err
	}
	return partial.Size(i)
}

// Manifest returns this image's Manifest object.
func (i *image) Manifest() (*v1.Manifest, error) {
	if err := i.compute(); err != nil {
		return nil, err
	}
	return i.manifest.DeepCopy(), nil
}

// RawManifest returns the serialized bytes of Manifest()
func (i *image) RawManifest() ([]byte, error) {
	if err := i.compute(); err != nil {
		return nil, err
	}
	return json.Marshal(i.manifest)
}

// LayerByDigest returns a Layer for interacting with a particular layer of
// the image, looking it up by "digest" (the compressed hash).
func (i *image) LayerByDigest(h v1.Hash) (v1.Layer, error) {
	if cn, err := i.ConfigName(); err != nil {
		return nil, err
	} else if h == cn {
		return partial.ConfigLayer(i)
	}
	if layer, ok := i.digestMap[h]; ok {
		return layer, nil
	}
	return i.base.LayerByDigest(h)
}

// LayerByDiffID is an analog to LayerByDigest, looking up by "diff id"
// (the uncompressed hash).
func (i *image) LayerByDiffID(h v1.Hash) (v1.Layer, error) {
	if layer, ok := i.diffIDMap[h]; ok {
		return layer, nil
	}
	return i.base.LayerByDiffID(h)
}

func validate(adds []Addendum) error {
	for _, add := range adds {
		if add.Layer == nil && !add.History.EmptyLayer {
			return errors.New("unable to add a nil layer to the image")
		}
	}
	return nil
}
