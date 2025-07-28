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
	"io"
	"net/http"
	"net/url"
	"sync"

	"github.com/google/go-containerregistry/internal/redact"
	"github.com/google/go-containerregistry/internal/verify"
	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/remote/transport"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

var acceptableImageMediaTypes = []types.MediaType{
	types.DockerManifestSchema2,
	types.OCIManifestSchema1,
}

// remoteImage accesses an image from a remote registry
type remoteImage struct {
	fetcher      fetcher
	ref          name.Reference
	ctx          context.Context
	manifestLock sync.Mutex // Protects manifest
	manifest     []byte
	configLock   sync.Mutex // Protects config
	config       []byte
	mediaType    types.MediaType
	descriptor   *v1.Descriptor
}

func (r *remoteImage) ArtifactType() (string, error) {
	// kind of a hack, but RawManifest does appropriate locking/memoization
	// and makes sure r.descriptor is populated.
	if _, err := r.RawManifest(); err != nil {
		return "", err
	}
	return r.descriptor.ArtifactType, nil
}

var _ partial.CompressedImageCore = (*remoteImage)(nil)

// Image provides access to a remote image reference.
func Image(ref name.Reference, options ...Option) (v1.Image, error) {
	desc, err := Get(ref, options...)
	if err != nil {
		return nil, err
	}

	return desc.Image()
}

func (r *remoteImage) MediaType() (types.MediaType, error) {
	if string(r.mediaType) != "" {
		return r.mediaType, nil
	}
	return types.DockerManifestSchema2, nil
}

func (r *remoteImage) RawManifest() ([]byte, error) {
	r.manifestLock.Lock()
	defer r.manifestLock.Unlock()
	if r.manifest != nil {
		return r.manifest, nil
	}

	// NOTE(jonjohnsonjr): We should never get here because the public entrypoints
	// do type-checking via remote.Descriptor. I've left this here for tests that
	// directly instantiate a remoteImage.
	manifest, desc, err := r.fetcher.fetchManifest(r.ctx, r.ref, acceptableImageMediaTypes)
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

func (r *remoteImage) RawConfigFile() ([]byte, error) {
	r.configLock.Lock()
	defer r.configLock.Unlock()
	if r.config != nil {
		return r.config, nil
	}

	m, err := partial.Manifest(r)
	if err != nil {
		return nil, err
	}

	if m.Config.Data != nil {
		if err := verify.Descriptor(m.Config); err != nil {
			return nil, err
		}
		r.config = m.Config.Data
		return r.config, nil
	}

	body, err := r.fetcher.fetchBlob(r.ctx, m.Config.Size, m.Config.Digest)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	r.config, err = io.ReadAll(body)
	if err != nil {
		return nil, err
	}
	return r.config, nil
}

// Descriptor retains the original descriptor from an index manifest.
// See partial.Descriptor.
func (r *remoteImage) Descriptor() (*v1.Descriptor, error) {
	// kind of a hack, but RawManifest does appropriate locking/memoization
	// and makes sure r.descriptor is populated.
	_, err := r.RawManifest()
	return r.descriptor, err
}

func (r *remoteImage) ConfigLayer() (v1.Layer, error) {
	if _, err := r.RawManifest(); err != nil {
		return nil, err
	}
	m, err := partial.Manifest(r)
	if err != nil {
		return nil, err
	}

	return partial.CompressedToLayer(&remoteImageLayer{
		ri:     r,
		ctx:    r.ctx,
		digest: m.Config.Digest,
	})
}

// remoteImageLayer implements partial.CompressedLayer
type remoteImageLayer struct {
	ri     *remoteImage
	ctx    context.Context
	digest v1.Hash
}

// Digest implements partial.CompressedLayer
func (rl *remoteImageLayer) Digest() (v1.Hash, error) {
	return rl.digest, nil
}

// Compressed implements partial.CompressedLayer
func (rl *remoteImageLayer) Compressed() (io.ReadCloser, error) {
	urls := []url.URL{rl.ri.fetcher.url("blobs", rl.digest.String())}

	// Add alternative layer sources from URLs (usually none).
	d, err := partial.BlobDescriptor(rl, rl.digest)
	if err != nil {
		return nil, err
	}

	if d.Data != nil {
		return verify.ReadCloser(io.NopCloser(bytes.NewReader(d.Data)), d.Size, d.Digest)
	}

	// We don't want to log binary layers -- this can break terminals.
	ctx := redact.NewContext(rl.ctx, "omitting binary blobs from logs")

	for _, s := range d.URLs {
		u, err := url.Parse(s)
		if err != nil {
			return nil, err
		}
		urls = append(urls, *u)
	}

	// The lastErr for most pulls will be the same (the first error), but for
	// foreign layers we'll want to surface the last one, since we try to pull
	// from the registry first, which would often fail.
	// TODO: Maybe we don't want to try pulling from the registry first?
	var lastErr error
	for _, u := range urls {
		req, err := http.NewRequest(http.MethodGet, u.String(), nil)
		if err != nil {
			return nil, err
		}

		resp, err := rl.ri.fetcher.Do(req.WithContext(ctx))
		if err != nil {
			lastErr = err
			continue
		}

		if err := transport.CheckError(resp, http.StatusOK); err != nil {
			resp.Body.Close()
			lastErr = err
			continue
		}

		return verify.ReadCloser(resp.Body, d.Size, rl.digest)
	}

	return nil, lastErr
}

// Manifest implements partial.WithManifest so that we can use partial.BlobSize below.
func (rl *remoteImageLayer) Manifest() (*v1.Manifest, error) {
	return partial.Manifest(rl.ri)
}

// MediaType implements v1.Layer
func (rl *remoteImageLayer) MediaType() (types.MediaType, error) {
	bd, err := partial.BlobDescriptor(rl, rl.digest)
	if err != nil {
		return "", err
	}

	return bd.MediaType, nil
}

// Size implements partial.CompressedLayer
func (rl *remoteImageLayer) Size() (int64, error) {
	// Look up the size of this digest in the manifest to avoid a request.
	return partial.BlobSize(rl, rl.digest)
}

// ConfigFile implements partial.WithManifestAndConfigFile so that we can use partial.BlobToDiffID below.
func (rl *remoteImageLayer) ConfigFile() (*v1.ConfigFile, error) {
	return partial.ConfigFile(rl.ri)
}

// DiffID implements partial.WithDiffID so that we don't recompute a DiffID that we already have
// available in our ConfigFile.
func (rl *remoteImageLayer) DiffID() (v1.Hash, error) {
	return partial.BlobToDiffID(rl, rl.digest)
}

// Descriptor retains the original descriptor from an image manifest.
// See partial.Descriptor.
func (rl *remoteImageLayer) Descriptor() (*v1.Descriptor, error) {
	return partial.BlobDescriptor(rl, rl.digest)
}

// See partial.Exists.
func (rl *remoteImageLayer) Exists() (bool, error) {
	return rl.ri.fetcher.blobExists(rl.ri.ctx, rl.digest)
}

// LayerByDigest implements partial.CompressedLayer
func (r *remoteImage) LayerByDigest(h v1.Hash) (partial.CompressedLayer, error) {
	return &remoteImageLayer{
		ri:     r,
		ctx:    r.ctx,
		digest: h,
	}, nil
}
