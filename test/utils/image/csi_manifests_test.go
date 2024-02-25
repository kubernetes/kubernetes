/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package image

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestCSIImageConfigs(t *testing.T) {
	configs := map[ImageID]Config{}
	appendCSIImageConfigs(configs)

	// We expect at least one entry for each of these images. There may be
	// more than one entry for the same image when different YAMLs use
	// different versions.
	//
	// The exact versions are not checked here because that would bring
	// back the problem of updating the expected versions. The set of
	// images shouldn't change much.
	expectedImages := []string{
		"csi-attacher",
		"csi-external-health-monitor-controller",
		"csi-node-driver-registrar",
		"csi-provisioner",
		"csi-resizer",
		"csi-snapshotter",
		"hostpathplugin",
		"livenessprobe",

		// From the GCP deployment.
		"gcp-compute-persistent-disk-csi-driver",

		// For some hostpath tests.
		"busybox",

		// For AnyVolumeDataSource feature tests.
		"volume-data-source-validator",
		"hello-populator",
	}
	actualImages := sets.NewString()
	for _, config := range configs {
		assert.NotEmpty(t, config.registry, "registry")
		assert.NotEmpty(t, config.name, "name")
		assert.NotEmpty(t, config.version, "version")
		actualImages.Insert(config.name)
	}
	assert.ElementsMatch(t, expectedImages, actualImages.UnsortedList(), "found these images: %+v", configs)
}
