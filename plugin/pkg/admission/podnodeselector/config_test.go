/*
Copyright 2017 The Kubernetes Authors.

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

package podnodeselector

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/labels"
)

const apiGroup = "podnodeselector.admission.k8s.io"

var supportedConfigAPIVersions []string = []string{"v1"}

// TestConfigParse tests config parsing for versioned configuration data
func TestConfigParse(t *testing.T) {
	for _, v := range supportedConfigAPIVersions {
		testName := fmt.Sprintf("Test config parse for api version: %s", v)
		version := apiGroup + "/" + v

		t.Run(testName, func(t *testing.T) {
			content := `apiVersion: ` + version + `
kind: Configuration
clusterDefaultNodeSelectors: "env=development"
namespaceSelectorsWhitelists:
  testNamespace1: "env=production"
  testNamespace2: "env=integration"`

			parsed, err := loadConfiguration(bytes.NewBufferString(content))
			if err != nil {
				t.Error("encountered unexpected error when loading configuration", err)
			}

			expectedClusterDefaultNodeSelectors := labels.Set{"env": "development"}
			expectedNamespaceSelectorsWhitelists := map[string]labels.Set{
				"testNamespace1": {"env": "production"},
				"testNamespace2": {"env": "integration"},
			}

			// parsing from label map -> labels.Set happens in admin controller ctor
			// go ahead and do that here so we test/validate the parsing as well
			nps, err := NewPodNodeSelector(parsed)
			if err != nil {
				t.Fatalf("unexpected error when constructing the pod node selector")
			}

			if !reflect.DeepEqual(expectedClusterDefaultNodeSelectors, nps.clusterDefaultNodeSelectors) {
				t.Errorf("failed to parse expected clusterDefaultNodeSelectors. expected: %#v+. got: %#v+.",
					expectedClusterDefaultNodeSelectors, parsed.ClusterDefaultNodeSelectors)
			}

			if !reflect.DeepEqual(expectedNamespaceSelectorsWhitelists, nps.namespaceSelectorsWhitelists) {
				t.Errorf("failed to parse expected namespaceSelectorsWhitelists. expected: %s. got: %s.",
					expectedNamespaceSelectorsWhitelists, parsed.NamespaceSelectorsWhitelists)
			}
		})
	}
}
