// Copyright 2017 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package acitest provides utilities for ACI testing.
package acitest

import (
	"encoding/json"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

var (
	// zeroVersion specifies an empty instance of the semantic version
	// used to compare it in the image manifest in order to detect an
	// unspecified manifest version.
	zeroVersion types.SemVer
)

const (
	// defaultName defines the name of the image manifest that will be
	// used when name was not specified.
	defaultName = "example.com/test01"
)

type (
	// These types are aliases to the types used in the application
	// container definition. They used to bypass validations of the
	// image manifest during serialization to JSON string.
	aciApp          types.App
	aciAnnotations  types.Annotations
	aciDependencies types.Dependencies
	aciKind         types.ACKind
	aciLabels       types.Labels
	aciName         types.ACIdentifier

	// aciManifest completely copies and underlying representation
	// of the application image manifest. It wraps each attribute of
	// the manifest into the alias in order to bypass validations.
	aciManifest struct {
		ACKind        aciKind         `json:"acKind"`
		ACVersion     types.SemVer    `json:"acVersion"`
		Name          aciName         `json:"name"`
		Labels        aciLabels       `json:"labels,omitempty"`
		App           *aciApp         `json:"app,omitempty"`
		Annotations   aciAnnotations  `json:"annotations,omitempty"`
		Dependencies  aciDependencies `json:"dependencies,omitempty"`
		PathWhitelist []string        `json:"pathWhitelist,omitempty"`
	}
)

// toImageManifest converts the specified application image manifest
// into the non-checkable image manifest, thus make it possible to
// create a manifest with mistakes.
func toImageManifest(m *schema.ImageManifest) *aciManifest {
	return &aciManifest{
		ACKind:        aciKind(m.ACKind),
		ACVersion:     m.ACVersion,
		Name:          aciName(m.Name),
		Labels:        aciLabels(m.Labels),
		App:           (*aciApp)(m.App),
		Annotations:   aciAnnotations(m.Annotations),
		Dependencies:  aciDependencies(m.Dependencies),
		PathWhitelist: m.PathWhitelist,
	}
}

// ImageManifestString sets the default attributes to the specified image
// manifest and returns its JSON string representation.
func ImageManifestString(im *schema.ImageManifest) (string, error) {
	// When nil have been passed as an argument, it will create an
	// empty manifest and define the minimal attributes.
	if im == nil {
		im = new(schema.ImageManifest)
	}

	// Set the default kind of the AC image manifest.
	if im.ACKind == "" {
		im.ACKind = schema.ImageManifestKind
	}

	// Set the default version of the AC image manifest.
	if im.ACVersion == zeroVersion {
		im.ACVersion = schema.AppContainerVersion
	}

	// Set the default name of the AC image manifest.
	if im.Name == "" {
		im.Name = defaultName
	}

	b, err := json.Marshal(toImageManifest(im))
	return string(b), err
}
