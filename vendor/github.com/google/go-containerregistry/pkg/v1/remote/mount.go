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

package remote

import (
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
)

// MountableLayer wraps a v1.Layer in a shim that enables the layer to be
// "mounted" when published to another registry.
type MountableLayer struct {
	v1.Layer

	Reference name.Reference
}

// Descriptor retains the original descriptor from an image manifest.
// See partial.Descriptor.
func (ml *MountableLayer) Descriptor() (*v1.Descriptor, error) {
	return partial.Descriptor(ml.Layer)
}

// Exists is a hack. See partial.Exists.
func (ml *MountableLayer) Exists() (bool, error) {
	return partial.Exists(ml.Layer)
}

// mountableImage wraps the v1.Layer references returned by the embedded v1.Image
// in MountableLayer's so that remote.Write might attempt to mount them from their
// source repository.
type mountableImage struct {
	v1.Image

	Reference name.Reference
}

// Layers implements v1.Image
func (mi *mountableImage) Layers() ([]v1.Layer, error) {
	ls, err := mi.Image.Layers()
	if err != nil {
		return nil, err
	}
	mls := make([]v1.Layer, 0, len(ls))
	for _, l := range ls {
		mls = append(mls, &MountableLayer{
			Layer:     l,
			Reference: mi.Reference,
		})
	}
	return mls, nil
}

// LayerByDigest implements v1.Image
func (mi *mountableImage) LayerByDigest(d v1.Hash) (v1.Layer, error) {
	l, err := mi.Image.LayerByDigest(d)
	if err != nil {
		return nil, err
	}
	return &MountableLayer{
		Layer:     l,
		Reference: mi.Reference,
	}, nil
}

// LayerByDiffID implements v1.Image
func (mi *mountableImage) LayerByDiffID(d v1.Hash) (v1.Layer, error) {
	l, err := mi.Image.LayerByDiffID(d)
	if err != nil {
		return nil, err
	}
	return &MountableLayer{
		Layer:     l,
		Reference: mi.Reference,
	}, nil
}

// Descriptor retains the original descriptor from an index manifest.
// See partial.Descriptor.
func (mi *mountableImage) Descriptor() (*v1.Descriptor, error) {
	return partial.Descriptor(mi.Image)
}

// ConfigLayer retains the original reference so that it can be mounted.
// See partial.ConfigLayer.
func (mi *mountableImage) ConfigLayer() (v1.Layer, error) {
	l, err := partial.ConfigLayer(mi.Image)
	if err != nil {
		return nil, err
	}
	return &MountableLayer{
		Layer:     l,
		Reference: mi.Reference,
	}, nil
}
