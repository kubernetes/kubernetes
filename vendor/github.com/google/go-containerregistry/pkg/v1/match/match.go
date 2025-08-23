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

// Package match provides functionality for conveniently matching a v1.Descriptor.
package match

import (
	v1 "github.com/google/go-containerregistry/pkg/v1"
	imagespec "github.com/opencontainers/image-spec/specs-go/v1"
)

// Matcher function that is given a v1.Descriptor, and returns whether or
// not it matches a given rule. Can match on anything it wants in the Descriptor.
type Matcher func(desc v1.Descriptor) bool

// Name returns a match.Matcher that matches based on the value of the
//
//	"org.opencontainers.image.ref.name" annotation:
//
// github.com/opencontainers/image-spec/blob/v1.0.1/annotations.md#pre-defined-annotation-keys
func Name(name string) Matcher {
	return Annotation(imagespec.AnnotationRefName, name)
}

// Annotation returns a match.Matcher that matches based on the provided annotation.
func Annotation(key, value string) Matcher {
	return func(desc v1.Descriptor) bool {
		if desc.Annotations == nil {
			return false
		}
		if aValue, ok := desc.Annotations[key]; ok && aValue == value {
			return true
		}
		return false
	}
}

// Platforms returns a match.Matcher that matches on any one of the provided platforms.
// Ignores any descriptors that do not have a platform.
func Platforms(platforms ...v1.Platform) Matcher {
	return func(desc v1.Descriptor) bool {
		if desc.Platform == nil {
			return false
		}
		for _, platform := range platforms {
			if desc.Platform.Equals(platform) {
				return true
			}
		}
		return false
	}
}

// MediaTypes returns a match.Matcher that matches at least one of the provided media types.
func MediaTypes(mediaTypes ...string) Matcher {
	mts := map[string]bool{}
	for _, media := range mediaTypes {
		mts[media] = true
	}
	return func(desc v1.Descriptor) bool {
		if desc.MediaType == "" {
			return false
		}
		if _, ok := mts[string(desc.MediaType)]; ok {
			return true
		}
		return false
	}
}

// Digests returns a match.Matcher that matches at least one of the provided Digests
func Digests(digests ...v1.Hash) Matcher {
	digs := map[v1.Hash]bool{}
	for _, digest := range digests {
		digs[digest] = true
	}
	return func(desc v1.Descriptor) bool {
		_, ok := digs[desc.Digest]
		return ok
	}
}
