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
