// Copyright 2020 Google LLC All Rights Reserved.
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
	"fmt"

	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/match"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// FindManifests given a v1.ImageIndex, find the manifests that fit the matcher.
func FindManifests(index v1.ImageIndex, matcher match.Matcher) ([]v1.Descriptor, error) {
	// get the actual manifest list
	indexManifest, err := index.IndexManifest()
	if err != nil {
		return nil, fmt.Errorf("unable to get raw index: %w", err)
	}
	manifests := []v1.Descriptor{}
	// try to get the root of our image
	for _, manifest := range indexManifest.Manifests {
		if matcher(manifest) {
			manifests = append(manifests, manifest)
		}
	}
	return manifests, nil
}

// FindImages given a v1.ImageIndex, find the images that fit the matcher. If a Descriptor
// matches the provider Matcher, but the referenced item is not an Image, ignores it.
// Only returns those that match the Matcher and are images.
func FindImages(index v1.ImageIndex, matcher match.Matcher) ([]v1.Image, error) {
	matches := []v1.Image{}
	manifests, err := FindManifests(index, matcher)
	if err != nil {
		return nil, err
	}
	for _, desc := range manifests {
		// if it is not an image, ignore it
		if !desc.MediaType.IsImage() {
			continue
		}
		img, err := index.Image(desc.Digest)
		if err != nil {
			return nil, err
		}
		matches = append(matches, img)
	}
	return matches, nil
}

// FindIndexes given a v1.ImageIndex, find the indexes that fit the matcher. If a Descriptor
// matches the provider Matcher, but the referenced item is not an Index, ignores it.
// Only returns those that match the Matcher and are indexes.
func FindIndexes(index v1.ImageIndex, matcher match.Matcher) ([]v1.ImageIndex, error) {
	matches := []v1.ImageIndex{}
	manifests, err := FindManifests(index, matcher)
	if err != nil {
		return nil, err
	}
	for _, desc := range manifests {
		if !desc.MediaType.IsIndex() {
			continue
		}
		// if it is not an index, ignore it
		idx, err := index.ImageIndex(desc.Digest)
		if err != nil {
			return nil, err
		}
		matches = append(matches, idx)
	}
	return matches, nil
}

type withManifests interface {
	Manifests() ([]Describable, error)
}

type withLayer interface {
	Layer(v1.Hash) (v1.Layer, error)
}

type describable struct {
	desc v1.Descriptor
}

func (d describable) Digest() (v1.Hash, error) {
	return d.desc.Digest, nil
}

func (d describable) Size() (int64, error) {
	return d.desc.Size, nil
}

func (d describable) MediaType() (types.MediaType, error) {
	return d.desc.MediaType, nil
}

func (d describable) Descriptor() (*v1.Descriptor, error) {
	return &d.desc, nil
}

// Manifests is analogous to v1.Image.Layers in that it allows values in the
// returned list to be lazily evaluated, which enables an index to contain
// an image that contains a streaming layer.
//
// This should have been part of the v1.ImageIndex interface, but wasn't.
// It is instead usable through this extension interface.
func Manifests(idx v1.ImageIndex) ([]Describable, error) {
	if wm, ok := idx.(withManifests); ok {
		return wm.Manifests()
	}

	return ComputeManifests(idx)
}

// ComputeManifests provides a fallback implementation for Manifests.
func ComputeManifests(idx v1.ImageIndex) ([]Describable, error) {
	m, err := idx.IndexManifest()
	if err != nil {
		return nil, err
	}
	manifests := []Describable{}
	for _, desc := range m.Manifests {
		switch {
		case desc.MediaType.IsImage():
			img, err := idx.Image(desc.Digest)
			if err != nil {
				return nil, err
			}
			manifests = append(manifests, img)
		case desc.MediaType.IsIndex():
			idx, err := idx.ImageIndex(desc.Digest)
			if err != nil {
				return nil, err
			}
			manifests = append(manifests, idx)
		default:
			if wl, ok := idx.(withLayer); ok {
				layer, err := wl.Layer(desc.Digest)
				if err != nil {
					return nil, err
				}
				manifests = append(manifests, layer)
			} else {
				manifests = append(manifests, describable{desc})
			}
		}
	}

	return manifests, nil
}
