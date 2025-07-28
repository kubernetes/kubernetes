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

// Package types holds common OCI media types.
package types

// MediaType is an enumeration of the supported mime types that an element of an image might have.
type MediaType string

// The collection of known MediaType values.
const (
	OCIContentDescriptor           MediaType = "application/vnd.oci.descriptor.v1+json"
	OCIImageIndex                  MediaType = "application/vnd.oci.image.index.v1+json"
	OCIManifestSchema1             MediaType = "application/vnd.oci.image.manifest.v1+json"
	OCIConfigJSON                  MediaType = "application/vnd.oci.image.config.v1+json"
	OCILayer                       MediaType = "application/vnd.oci.image.layer.v1.tar+gzip"
	OCILayerZStd                   MediaType = "application/vnd.oci.image.layer.v1.tar+zstd"
	OCIRestrictedLayer             MediaType = "application/vnd.oci.image.layer.nondistributable.v1.tar+gzip"
	OCIUncompressedLayer           MediaType = "application/vnd.oci.image.layer.v1.tar"
	OCIUncompressedRestrictedLayer MediaType = "application/vnd.oci.image.layer.nondistributable.v1.tar"

	DockerManifestSchema1       MediaType = "application/vnd.docker.distribution.manifest.v1+json"
	DockerManifestSchema1Signed MediaType = "application/vnd.docker.distribution.manifest.v1+prettyjws"
	DockerManifestSchema2       MediaType = "application/vnd.docker.distribution.manifest.v2+json"
	DockerManifestList          MediaType = "application/vnd.docker.distribution.manifest.list.v2+json"
	DockerLayer                 MediaType = "application/vnd.docker.image.rootfs.diff.tar.gzip"
	DockerConfigJSON            MediaType = "application/vnd.docker.container.image.v1+json"
	DockerPluginConfig          MediaType = "application/vnd.docker.plugin.v1+json"
	DockerForeignLayer          MediaType = "application/vnd.docker.image.rootfs.foreign.diff.tar.gzip"
	DockerUncompressedLayer     MediaType = "application/vnd.docker.image.rootfs.diff.tar"

	OCIVendorPrefix    = "vnd.oci"
	DockerVendorPrefix = "vnd.docker"
)

// IsDistributable returns true if a layer is distributable, see:
// https://github.com/opencontainers/image-spec/blob/master/layer.md#non-distributable-layers
func (m MediaType) IsDistributable() bool {
	switch m {
	case DockerForeignLayer, OCIRestrictedLayer, OCIUncompressedRestrictedLayer:
		return false
	}
	return true
}

// IsImage returns true if the mediaType represents an image manifest, as opposed to something else, like an index.
func (m MediaType) IsImage() bool {
	switch m {
	case OCIManifestSchema1, DockerManifestSchema2:
		return true
	}
	return false
}

// IsIndex returns true if the mediaType represents an index, as opposed to something else, like an image.
func (m MediaType) IsIndex() bool {
	switch m {
	case OCIImageIndex, DockerManifestList:
		return true
	}
	return false
}

// IsConfig returns true if the mediaType represents a config, as opposed to something else, like an image.
func (m MediaType) IsConfig() bool {
	switch m {
	case OCIConfigJSON, DockerConfigJSON:
		return true
	}
	return false
}

func (m MediaType) IsSchema1() bool {
	switch m {
	case DockerManifestSchema1, DockerManifestSchema1Signed:
		return true
	}
	return false
}

func (m MediaType) IsLayer() bool {
	switch m {
	case DockerLayer, DockerUncompressedLayer, OCILayer, OCILayerZStd, OCIUncompressedLayer, DockerForeignLayer, OCIRestrictedLayer, OCIUncompressedRestrictedLayer:
		return true
	}
	return false
}
