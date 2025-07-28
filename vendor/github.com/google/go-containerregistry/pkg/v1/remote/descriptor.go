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
	"context"
	"errors"
	"fmt"

	"github.com/google/go-containerregistry/pkg/logs"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

var allManifestMediaTypes = append(append([]types.MediaType{
	types.DockerManifestSchema1,
	types.DockerManifestSchema1Signed,
}, acceptableImageMediaTypes...), acceptableIndexMediaTypes...)

// ErrSchema1 indicates that we received a schema1 manifest from the registry.
// This library doesn't have plans to support this legacy image format:
// https://github.com/google/go-containerregistry/issues/377
var ErrSchema1 = errors.New("see https://github.com/google/go-containerregistry/issues/377")

// newErrSchema1 returns an ErrSchema1 with the unexpected MediaType.
func newErrSchema1(schema types.MediaType) error {
	return fmt.Errorf("unsupported MediaType: %q, %w", schema, ErrSchema1)
}

// Descriptor provides access to metadata about remote artifact and accessors
// for efficiently converting it into a v1.Image or v1.ImageIndex.
type Descriptor struct {
	fetcher fetcher
	v1.Descriptor

	ref      name.Reference
	Manifest []byte
	ctx      context.Context

	// So we can share this implementation with Image.
	platform v1.Platform
}

func (d *Descriptor) toDesc() v1.Descriptor {
	return d.Descriptor
}

// RawManifest exists to satisfy the Taggable interface.
func (d *Descriptor) RawManifest() ([]byte, error) {
	return d.Manifest, nil
}

// Get returns a remote.Descriptor for the given reference. The response from
// the registry is left un-interpreted, for the most part. This is useful for
// querying what kind of artifact a reference represents.
//
// See Head if you don't need the response body.
func Get(ref name.Reference, options ...Option) (*Descriptor, error) {
	return get(ref, allManifestMediaTypes, options...)
}

// Head returns a v1.Descriptor for the given reference by issuing a HEAD
// request.
//
// Note that the server response will not have a body, so any errors encountered
// should be retried with Get to get more details.
func Head(ref name.Reference, options ...Option) (*v1.Descriptor, error) {
	o, err := makeOptions(options...)
	if err != nil {
		return nil, err
	}

	return newPuller(o).Head(o.context, ref)
}

// Handle options and fetch the manifest with the acceptable MediaTypes in the
// Accept header.
func get(ref name.Reference, acceptable []types.MediaType, options ...Option) (*Descriptor, error) {
	o, err := makeOptions(options...)
	if err != nil {
		return nil, err
	}
	return newPuller(o).get(o.context, ref, acceptable, o.platform)
}

// Image converts the Descriptor into a v1.Image.
//
// If the fetched artifact is already an image, it will just return it.
//
// If the fetched artifact is an index, it will attempt to resolve the index to
// a child image with the appropriate platform.
//
// See WithPlatform to set the desired platform.
func (d *Descriptor) Image() (v1.Image, error) {
	switch d.MediaType {
	case types.DockerManifestSchema1, types.DockerManifestSchema1Signed:
		// We don't care to support schema 1 images:
		// https://github.com/google/go-containerregistry/issues/377
		return nil, newErrSchema1(d.MediaType)
	case types.OCIImageIndex, types.DockerManifestList:
		// We want an image but the registry has an index, resolve it to an image.
		return d.remoteIndex().imageByPlatform(d.platform)
	case types.OCIManifestSchema1, types.DockerManifestSchema2:
		// These are expected. Enumerated here to allow a default case.
	default:
		// We could just return an error here, but some registries (e.g. static
		// registries) don't set the Content-Type headers correctly, so instead...
		logs.Warn.Printf("Unexpected media type for Image(): %s", d.MediaType)
	}

	// Wrap the v1.Layers returned by this v1.Image in a hint for downstream
	// remote.Write calls to facilitate cross-repo "mounting".
	imgCore, err := partial.CompressedToImage(d.remoteImage())
	if err != nil {
		return nil, err
	}
	return &mountableImage{
		Image:     imgCore,
		Reference: d.ref,
	}, nil
}

// Schema1 converts the Descriptor into a v1.Image for v2 schema 1 media types.
//
// The v1.Image returned by this method does not implement the entire interface because it would be inefficient.
// This exists mostly to make it easier to copy schema 1 images around or look at their filesystems.
// This is separate from Image() to avoid a backward incompatible change for callers expecting ErrSchema1.
func (d *Descriptor) Schema1() (v1.Image, error) {
	i := &schema1{
		ref:        d.ref,
		fetcher:    d.fetcher,
		ctx:        d.ctx,
		manifest:   d.Manifest,
		mediaType:  d.MediaType,
		descriptor: &d.Descriptor,
	}

	return &mountableImage{
		Image:     i,
		Reference: d.ref,
	}, nil
}

// ImageIndex converts the Descriptor into a v1.ImageIndex.
func (d *Descriptor) ImageIndex() (v1.ImageIndex, error) {
	switch d.MediaType {
	case types.DockerManifestSchema1, types.DockerManifestSchema1Signed:
		// We don't care to support schema 1 images:
		// https://github.com/google/go-containerregistry/issues/377
		return nil, newErrSchema1(d.MediaType)
	case types.OCIManifestSchema1, types.DockerManifestSchema2:
		// We want an index but the registry has an image, nothing we can do.
		return nil, fmt.Errorf("unexpected media type for ImageIndex(): %s; call Image() instead", d.MediaType)
	case types.OCIImageIndex, types.DockerManifestList:
		// These are expected.
	default:
		// We could just return an error here, but some registries (e.g. static
		// registries) don't set the Content-Type headers correctly, so instead...
		logs.Warn.Printf("Unexpected media type for ImageIndex(): %s", d.MediaType)
	}
	return d.remoteIndex(), nil
}

func (d *Descriptor) remoteImage() *remoteImage {
	return &remoteImage{
		ref:        d.ref,
		ctx:        d.ctx,
		fetcher:    d.fetcher,
		manifest:   d.Manifest,
		mediaType:  d.MediaType,
		descriptor: &d.Descriptor,
	}
}

func (d *Descriptor) remoteIndex() *remoteIndex {
	return &remoteIndex{
		ref:        d.ref,
		ctx:        d.ctx,
		fetcher:    d.fetcher,
		manifest:   d.Manifest,
		mediaType:  d.MediaType,
		descriptor: &d.Descriptor,
	}
}
