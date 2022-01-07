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

package empty

import (
	"encoding/json"
	"errors"

	v1 "github.com/google/go-containerregistry/pkg/v1"
	"github.com/google/go-containerregistry/pkg/v1/partial"
	"github.com/google/go-containerregistry/pkg/v1/types"
)

// Index is a singleton empty index, think: FROM scratch.
var Index = emptyIndex{}

type emptyIndex struct{}

func (i emptyIndex) MediaType() (types.MediaType, error) {
	return types.OCIImageIndex, nil
}

func (i emptyIndex) Digest() (v1.Hash, error) {
	return partial.Digest(i)
}

func (i emptyIndex) Size() (int64, error) {
	return partial.Size(i)
}

func (i emptyIndex) IndexManifest() (*v1.IndexManifest, error) {
	return base(), nil
}

func (i emptyIndex) RawManifest() ([]byte, error) {
	return json.Marshal(base())
}

func (i emptyIndex) Image(v1.Hash) (v1.Image, error) {
	return nil, errors.New("empty index")
}

func (i emptyIndex) ImageIndex(v1.Hash) (v1.ImageIndex, error) {
	return nil, errors.New("empty index")
}

func base() *v1.IndexManifest {
	return &v1.IndexManifest{
		SchemaVersion: 2,
	}
}
