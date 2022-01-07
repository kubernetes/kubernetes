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

package v1

import (
	"encoding/json"
	"io"

	"github.com/google/go-containerregistry/pkg/v1/types"
)

// Manifest represents the OCI image manifest in a structured way.
type Manifest struct {
	SchemaVersion int64             `json:"schemaVersion"`
	MediaType     types.MediaType   `json:"mediaType,omitempty"`
	Config        Descriptor        `json:"config"`
	Layers        []Descriptor      `json:"layers"`
	Annotations   map[string]string `json:"annotations,omitempty"`
}

// IndexManifest represents an OCI image index in a structured way.
type IndexManifest struct {
	SchemaVersion int64             `json:"schemaVersion"`
	MediaType     types.MediaType   `json:"mediaType,omitempty"`
	Manifests     []Descriptor      `json:"manifests"`
	Annotations   map[string]string `json:"annotations,omitempty"`
}

// Descriptor holds a reference from the manifest to one of its constituent elements.
type Descriptor struct {
	MediaType   types.MediaType   `json:"mediaType"`
	Size        int64             `json:"size"`
	Digest      Hash              `json:"digest"`
	Data        []byte            `json:"data,omitempty"`
	URLs        []string          `json:"urls,omitempty"`
	Annotations map[string]string `json:"annotations,omitempty"`
	Platform    *Platform         `json:"platform,omitempty"`
}

// ParseManifest parses the io.Reader's contents into a Manifest.
func ParseManifest(r io.Reader) (*Manifest, error) {
	m := Manifest{}
	if err := json.NewDecoder(r).Decode(&m); err != nil {
		return nil, err
	}
	return &m, nil
}

// ParseIndexManifest parses the io.Reader's contents into an IndexManifest.
func ParseIndexManifest(r io.Reader) (*IndexManifest, error) {
	im := IndexManifest{}
	if err := json.NewDecoder(r).Decode(&im); err != nil {
		return nil, err
	}
	return &im, nil
}
