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
	"bytes"
	"fmt"
	"io/fs"
	"regexp"
	"strings"

	"k8s.io/apimachinery/pkg/util/yaml"
	e2etestingmanifests "k8s.io/kubernetes/test/e2e/testing-manifests"
)

// All of the image tags are of the format registry.k8s.io/sig-storage/hostpathplugin:v1.7.3.
var imageRE = regexp.MustCompile(`^(.*)/([^/:]*):(.*)$`)

// appendCSIImageConfigs extracts image repo, name and version from
// the YAML files under test/e2e/testing-manifests/storage-csi and
// creates new config entries  for them.
func appendCSIImageConfigs(configs map[ImageID]Config) {
	embeddedFS := e2etestingmanifests.GetE2ETestingManifestsFS().EmbeddedFS

	// We add our images with ImageID numbers that start after the highest existing number.
	index := ImageID(0)
	for i := range configs {
		if i > index {
			index = i
		}
	}

	err := fs.WalkDir(embeddedFS, "storage-csi", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, ".yaml") {
			return nil
		}
		data, err := embeddedFS.ReadFile(path)
		if err != nil {
			return err
		}

		// Split at the "---" separator before working on
		// individual item. Only works for .yaml.
		//
		// We need to split ourselves because we need access
		// to each original chunk of data for
		// runtime.DecodeInto. kubectl has its own
		// infrastructure for this, but that is a lot of code
		// with many dependencies.
		items := bytes.Split(data, []byte("\n---"))
		for i, item := range items {
			// We don't care what the actual type is. We just
			// unmarshal into generic maps and then look up images.
			var object interface{}
			if err := yaml.Unmarshal(item, &object); err != nil {
				return fmt.Errorf("decode item #%d in %s: %v",
					i, path, err)
			}

			// Will be called for all image strings.
			visit := func(value string) {
				parts := imageRE.FindStringSubmatch(value)
				if parts == nil {
					return
				}
				config := Config{parts[1], parts[2], parts[3]}
				for _, otherConfig := range configs {
					if otherConfig == config {
						return
					}
				}
				index++
				configs[index] = config
			}

			// These paths match plain Pods and more complex types
			// like Deployments.
			findStrings(object, visit, "spec", "containers", "image")
			findStrings(object, visit, "spec", "template", "spec", "containers", "image")

		}
		return nil

	})
	if err != nil {
		panic(err)
	}
}

// findStrings recursively decends into an object along a certain path.  Path
// elements are the named fields. If a field references a list, each of the
// list elements will be followed.
//
// Conceptually this is similar to a JSON path.
func findStrings(object interface{}, visit func(value string), path ...string) {
	if len(path) == 0 {
		// Found it. May or may not be a string, though.
		if object, ok := object.(string); ok {
			visit(object)
		}
		return
	}

	switch object := object.(type) {
	case []interface{}:
		// If we are in a list, check each entry.
		for _, child := range object {
			findStrings(child, visit, path...)
		}
	case map[string]interface{}:
		// Follow path if possible
		if child, ok := object[path[0]]; ok {
			findStrings(child, visit, path[1:]...)
		}
	}
}
