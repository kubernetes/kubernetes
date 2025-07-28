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
	"bytes"
	"context"
	"fmt"
	"sync"

	"github.com/google/go-containerregistry/internal/verify"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

var acceptableIndexMediaTypes = []types.MediaType{
	types.DockerManifestList,
	types.OCIImageIndex,
}

// remoteIndex accesses an index from a remote registry
type remoteIndex struct {
	fetcher      fetcher
	ref          name.Reference
	ctx          context.Context
	manifestLock sync.Mutex // Protects manifest
	manifest     []byte
	mediaType    types.MediaType
	descriptor   *v1.Descriptor
}

// Index provides access to a remote index reference.
func Index(ref name.Reference, options ...Option) (v1.ImageIndex, error) {
	desc, err := get(ref, acceptableIndexMediaTypes, options...)
	if err != nil {
		return nil, err
	}

	return desc.ImageIndex()
}

func (r *remoteIndex) MediaType() (types.MediaType, error) {
	if string(r.mediaType) != "" {
		return r.mediaType, nil
	}
	return types.DockerManifestList, nil
}

func (r *remoteIndex) Digest() (v1.Hash, error) {
	return partial.Digest(r)
}

func (r *remoteIndex) Size() (int64, error) {
	return partial.Size(r)
}

func (r *remoteIndex) RawManifest() ([]byte, error) {
	r.manifestLock.Lock()
	defer r.manifestLock.Unlock()
	if r.manifest != nil {
		return r.manifest, nil
	}

	// NOTE(jonjohnsonjr): We should never get here because the public entrypoints
	// do type-checking via remote.Descriptor. I've left this here for tests that
	// directly instantiate a remoteIndex.
	manifest, desc, err := r.fetcher.fetchManifest(r.ctx, r.ref, acceptableIndexMediaTypes)
	if err != nil {
		return nil, err
	}

	if r.descriptor == nil {
		r.descriptor = desc
	}
	r.mediaType = desc.MediaType
	r.manifest = manifest
	return r.manifest, nil
}

func (r *remoteIndex) IndexManifest() (*v1.IndexManifest, error) {
	b, err := r.RawManifest()
	if err != nil {
		return nil, err
	}
	return v1.ParseIndexManifest(bytes.NewReader(b))
}

func (r *remoteIndex) Image(h v1.Hash) (v1.Image, error) {
	desc, err := r.childByHash(h)
	if err != nil {
		return nil, err
	}

	// Descriptor.Image will handle coercing nested indexes into an Image.
	return desc.Image()
}

// Descriptor retains the original descriptor from an index manifest.
// See partial.Descriptor.
func (r *remoteIndex) Descriptor() (*v1.Descriptor, error) {
	// kind of a hack, but RawManifest does appropriate locking/memoization
	// and makes sure r.descriptor is populated.
	_, err := r.RawManifest()
	return r.descriptor, err
}

func (r *remoteIndex) ImageIndex(h v1.Hash) (v1.ImageIndex, error) {
	desc, err := r.childByHash(h)
	if err != nil {
		return nil, err
	}
	return desc.ImageIndex()
}

// Workaround for #819.
func (r *remoteIndex) Layer(h v1.Hash) (v1.Layer, error) {
	index, err := r.IndexManifest()
	if err != nil {
		return nil, err
	}
	for _, childDesc := range index.Manifests {
		if h == childDesc.Digest {
			l, err := partial.CompressedToLayer(&remoteLayer{
				fetcher: r.fetcher,
				ctx:     r.ctx,
				digest:  h,
			})
			if err != nil {
				return nil, err
			}
			return &MountableLayer{
				Layer:     l,
				Reference: r.ref.Context().Digest(h.String()),
			}, nil
		}
	}
	return nil, fmt.Errorf("layer not found: %s", h)
}

func (r *remoteIndex) imageByPlatform(platform v1.Platform) (v1.Image, error) {
	desc, err := r.childByPlatform(platform)
	if err != nil {
		return nil, err
	}

	// Descriptor.Image will handle coercing nested indexes into an Image.
	return desc.Image()
}

// This naively matches the first manifest with matching platform attributes.
//
// We should probably use this instead:
//
//	github.com/containerd/containerd/platforms
//
// But first we'd need to migrate to:
//
//	github.com/opencontainers/image-spec/specs-go/v1
func (r *remoteIndex) childByPlatform(platform v1.Platform) (*Descriptor, error) {
	index, err := r.IndexManifest()
	if err != nil {
		return nil, err
	}
	for _, childDesc := range index.Manifests {
		// If platform is missing from child descriptor, assume it's amd64/linux.
		p := defaultPlatform
		if childDesc.Platform != nil {
			p = *childDesc.Platform
		}

		if matchesPlatform(p, platform) {
			return r.childDescriptor(childDesc, platform)
		}
	}
	return nil, fmt.Errorf("no child with platform %+v in index %s", platform, r.ref)
}

func (r *remoteIndex) childByHash(h v1.Hash) (*Descriptor, error) {
	index, err := r.IndexManifest()
	if err != nil {
		return nil, err
	}
	for _, childDesc := range index.Manifests {
		if h == childDesc.Digest {
			return r.childDescriptor(childDesc, defaultPlatform)
		}
	}
	return nil, fmt.Errorf("no child with digest %s in index %s", h, r.ref)
}

// Convert one of this index's child's v1.Descriptor into a remote.Descriptor, with the given platform option.
func (r *remoteIndex) childDescriptor(child v1.Descriptor, platform v1.Platform) (*Descriptor, error) {
	ref := r.ref.Context().Digest(child.Digest.String())
	var (
		manifest []byte
		err      error
	)
	if child.Data != nil {
		if err := verify.Descriptor(child); err != nil {
			return nil, err
		}
		manifest = child.Data
	} else {
		manifest, _, err = r.fetcher.fetchManifest(r.ctx, ref, []types.MediaType{child.MediaType})
		if err != nil {
			return nil, err
		}
	}

	if child.MediaType.IsImage() {
		mf, _ := v1.ParseManifest(bytes.NewReader(manifest))
		// Failing to parse as a manifest should just be ignored.
		// The manifest might not be valid, and that's okay.
		if mf != nil && !mf.Config.MediaType.IsConfig() {
			child.ArtifactType = string(mf.Config.MediaType)
		}
	}

	return &Descriptor{
		ref:        ref,
		ctx:        r.ctx,
		fetcher:    r.fetcher,
		Manifest:   manifest,
		Descriptor: child,
		platform:   platform,
	}, nil
}

// matchesPlatform checks if the given platform matches the required platforms.
// The given platform matches the required platform if
// - architecture and OS are identical.
// - OS version and variant are identical if provided.
// - features and OS features of the required platform are subsets of those of the given platform.
func matchesPlatform(given, required v1.Platform) bool {
	// Required fields that must be identical.
	if given.Architecture != required.Architecture || given.OS != required.OS {
		return false
	}

	// Optional fields that may be empty, but must be identical if provided.
	if required.OSVersion != "" && given.OSVersion != required.OSVersion {
		return false
	}
	if required.Variant != "" && given.Variant != required.Variant {
		return false
	}

	// Verify required platform's features are a subset of given platform's features.
	if !isSubset(given.OSFeatures, required.OSFeatures) {
		return false
	}
	if !isSubset(given.Features, required.Features) {
		return false
	}

	return true
}

// isSubset checks if the required array of strings is a subset of the given lst.
func isSubset(lst, required []string) bool {
	set := make(map[string]bool)
	for _, value := range lst {
		set[value] = true
	}

	for _, value := range required {
		if _, ok := set[value]; !ok {
			return false
		}
	}

	return true
}
