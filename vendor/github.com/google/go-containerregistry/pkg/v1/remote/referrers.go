// Copyright 2023 Google LLC All Rights Reserved.
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
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/empty"
	"github.com/google/go-containerregistry/pkg/v1/mutate"
	"github.com/google/go-containerregistry/pkg/v1/remote/transport"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// Referrers returns a list of descriptors that refer to the given manifest digest.
//
// The subject manifest doesn't have to exist in the registry for there to be descriptors that refer to it.
func Referrers(d name.Digest, options ...Option) (v1.ImageIndex, error) {
	o, err := makeOptions(options...)
	if err != nil {
		return nil, err
	}
	return newPuller(o).referrers(o.context, d, o.filter)
}

// https://github.com/opencontainers/distribution-spec/blob/main/spec.md#referrers-tag-schema
func fallbackTag(d name.Digest) name.Tag {
	return d.Context().Tag(strings.Replace(d.DigestStr(), ":", "-", 1))
}

func (f *fetcher) fetchReferrers(ctx context.Context, filter map[string]string, d name.Digest) (v1.ImageIndex, error) {
	// Check the Referrers API endpoint first.
	u := f.url("referrers", d.DigestStr())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", string(types.OCIImageIndex))

	resp, err := f.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if err := transport.CheckError(resp, http.StatusOK, http.StatusNotFound, http.StatusBadRequest, http.StatusNotAcceptable); err != nil {
		return nil, err
	}

	var b []byte
	if resp.StatusCode == http.StatusOK && resp.Header.Get("Content-Type") == string(types.OCIImageIndex) {
		b, err = io.ReadAll(resp.Body)
		if err != nil {
			return nil, err
		}
	} else {
		// The registry doesn't support the Referrers API endpoint, so we'll use the fallback tag scheme.
		b, _, err = f.fetchManifest(ctx, fallbackTag(d), []types.MediaType{types.OCIImageIndex})
		var terr *transport.Error
		if errors.As(err, &terr) && terr.StatusCode == http.StatusNotFound {
			// Not found just means there are no attachments yet. Start with an empty manifest.
			return empty.Index, nil
		} else if err != nil {
			return nil, err
		}
	}

	h, sz, err := v1.SHA256(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	idx := &remoteIndex{
		fetcher:   *f,
		ctx:       ctx,
		manifest:  b,
		mediaType: types.OCIImageIndex,
		descriptor: &v1.Descriptor{
			Digest:    h,
			MediaType: types.OCIImageIndex,
			Size:      sz,
		},
	}
	return filterReferrersResponse(filter, idx), nil
}

// If filter applied, filter out by artifactType.
// See https://github.com/opencontainers/distribution-spec/blob/main/spec.md#listing-referrers
func filterReferrersResponse(filter map[string]string, in v1.ImageIndex) v1.ImageIndex {
	if filter == nil {
		return in
	}
	v, ok := filter["artifactType"]
	if !ok {
		return in
	}
	return mutate.RemoveManifests(in, func(desc v1.Descriptor) bool {
		return desc.ArtifactType != v
	})
}
