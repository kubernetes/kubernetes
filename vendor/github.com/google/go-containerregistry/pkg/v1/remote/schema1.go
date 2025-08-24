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
	"encoding/json"

	"github.com/google/go-containerregistry/pkg/name"
	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

type schema1 struct {
	ref        name.Reference
	ctx        context.Context
	fetcher    fetcher
	manifest   []byte
	mediaType  types.MediaType
	descriptor *v1.Descriptor
}

func (s *schema1) Layers() ([]v1.Layer, error) {
	m := schema1Manifest{}
	if err := json.NewDecoder(bytes.NewReader(s.manifest)).Decode(&m); err != nil {
		return nil, err
	}

	layers := []v1.Layer{}
	for i := len(m.FSLayers) - 1; i >= 0; i-- {
		fsl := m.FSLayers[i]

		h, err := v1.NewHash(fsl.BlobSum)
		if err != nil {
			return nil, err
		}
		l, err := s.LayerByDigest(h)
		if err != nil {
			return nil, err
		}
		layers = append(layers, l)
	}

	return layers, nil
}

func (s *schema1) MediaType() (types.MediaType, error) {
	return s.mediaType, nil
}

func (s *schema1) Size() (int64, error) {
	return s.descriptor.Size, nil
}

func (s *schema1) ConfigName() (v1.Hash, error) {
	return partial.ConfigName(s)
}

func (s *schema1) ConfigFile() (*v1.ConfigFile, error) {
	return nil, newErrSchema1(s.mediaType)
}

func (s *schema1) RawConfigFile() ([]byte, error) {
	return []byte("{}"), nil
}

func (s *schema1) Digest() (v1.Hash, error) {
	return s.descriptor.Digest, nil
}

func (s *schema1) Manifest() (*v1.Manifest, error) {
	return nil, newErrSchema1(s.mediaType)
}

func (s *schema1) RawManifest() ([]byte, error) {
	return s.manifest, nil
}

func (s *schema1) LayerByDigest(h v1.Hash) (v1.Layer, error) {
	l, err := partial.CompressedToLayer(&remoteLayer{
		fetcher: s.fetcher,
		ctx:     s.ctx,
		digest:  h,
	})
	if err != nil {
		return nil, err
	}
	return &MountableLayer{
		Layer:     l,
		Reference: s.ref.Context().Digest(h.String()),
	}, nil
}

func (s *schema1) LayerByDiffID(v1.Hash) (v1.Layer, error) {
	return nil, newErrSchema1(s.mediaType)
}

type fslayer struct {
	BlobSum string `json:"blobSum"`
}

type schema1Manifest struct {
	FSLayers []fslayer `json:"fsLayers"`
}
