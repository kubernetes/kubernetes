// Copyright 2016 The Linux Foundation
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

package v1

import "github.com/opencontainers/image-spec/specs-go"

// Platform describes the platform which the image in the manifest runs on.
type Platform struct {
	// Architecture field specifies the CPU architecture, for example
	// `amd64` or `ppc64`.
	Architecture string `json:"architecture"`

	// OS specifies the operating system, for example `linux` or `windows`.
	OS string `json:"os"`

	// OSVersion is an optional field specifying the operating system
	// version, for example `10.0.10586`.
	OSVersion string `json:"os.version,omitempty"`

	// OSFeatures is an optional field specifying an array of strings,
	// each listing a required OS feature (for example on Windows `win32k`).
	OSFeatures []string `json:"os.features,omitempty"`

	// Variant is an optional field specifying a variant of the CPU, for
	// example `ppc64le` to specify a little-endian version of a PowerPC CPU.
	Variant string `json:"variant,omitempty"`

	// Features is an optional field specifying an array of strings, each
	// listing a required CPU feature (for example `sse4` or `aes`).
	Features []string `json:"features,omitempty"`
}

// ManifestDescriptor describes a platform specific manifest.
type ManifestDescriptor struct {
	Descriptor

	// Platform describes the platform which the image in the manifest runs on.
	Platform Platform `json:"platform"`
}

// ImageIndex references manifests for various platforms.
// This structure provides `application/vnd.oci.image.index.v1+json` mediatype when marshalled to JSON.
type ImageIndex struct {
	specs.Versioned

	// Manifests references platform specific manifests.
	Manifests []ManifestDescriptor `json:"manifests"`

	// Annotations contains arbitrary metadata for the image index.
	Annotations map[string]string `json:"annotations,omitempty"`
}
