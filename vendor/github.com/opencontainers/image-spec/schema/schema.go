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

package schema

import (
	"net/http"

	"github.com/opencontainers/image-spec/specs-go/v1"
)

// Media types for the OCI image formats
const (
	ValidatorMediaTypeDescriptor   Validator     = v1.MediaTypeDescriptor
	ValidatorMediaTypeLayoutHeader Validator     = v1.MediaTypeLayoutHeader
	ValidatorMediaTypeManifest     Validator     = v1.MediaTypeImageManifest
	ValidatorMediaTypeImageIndex   Validator     = v1.MediaTypeImageIndex
	ValidatorMediaTypeImageConfig  Validator     = v1.MediaTypeImageConfig
	ValidatorMediaTypeImageLayer   unimplemented = v1.MediaTypeImageLayer
)

var (
	// fs stores the embedded http.FileSystem
	// having the OCI JSON schema files in root "/".
	fs = _escFS(false)

	// specs maps OCI schema media types to schema files.
	specs = map[Validator]string{
		ValidatorMediaTypeDescriptor:   "content-descriptor.json",
		ValidatorMediaTypeLayoutHeader: "image-layout-schema.json",
		ValidatorMediaTypeManifest:     "image-manifest-schema.json",
		ValidatorMediaTypeImageIndex:   "image-index-schema.json",
		ValidatorMediaTypeImageConfig:  "config-schema.json",
	}
)

// FileSystem returns an in-memory filesystem including the schema files.
// The schema files are located at the root directory.
func FileSystem() http.FileSystem {
	return fs
}
