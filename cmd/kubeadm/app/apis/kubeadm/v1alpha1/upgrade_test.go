/*
Copyright 2018 The Kubernetes Authors.

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

package v1alpha1

import (
	"io/ioutil"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

const test196 = "testdata/kubeadm196.yaml"

func TestUpgrade(t *testing.T) {
	testYAML, err := ioutil.ReadFile(test196)
	if err != nil {
		t.Fatalf("couldn't read test data: %v", err)
	}

	decoded, err := LoadYAML(testYAML)
	if err != nil {
		t.Fatalf("couldn't unmarshal test yaml: %v", err)
	}

	var obj MasterConfiguration
	if err := Migrate(decoded, &obj); err != nil {
		t.Fatalf("couldn't decode migrated object: %v", err)
	}
}

func TestProxyFeatureListToMap(t *testing.T) {

	cases := []struct {
		name         string
		featureGates interface{}
		expected     map[string]interface{}
		shouldError  bool
	}{
		{
			name:         "multiple features",
			featureGates: "feature1=true,feature2=false",
			expected: map[string]interface{}{
				"feature1": true,
				"feature2": false,
			},
		},
		{
			name:         "single feature",
			featureGates: "feature1=true",
			expected: map[string]interface{}{
				"feature1": true,
			},
		},
		{
			name: "already a map",
			featureGates: map[string]interface{}{
				"feature1": true,
			},
			expected: map[string]interface{}{
				"feature1": true,
			},
		},
		{
			name:         "single feature",
			featureGates: "",
			expected:     map[string]interface{}{},
		},
		{
			name:         "malformed string",
			featureGates: "test,",
			shouldError:  true,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {

			cfg := map[string]interface{}{
				"kubeProxy": map[string]interface{}{
					"config": map[string]interface{}{
						"featureGates": testCase.featureGates,
					},
				},
			}

			err := proxyFeatureListToMap(cfg)
			if testCase.shouldError {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			gates, ok, err := unstructured.NestedMap(cfg, "kubeProxy", "config", "featureGates")
			if !ok {
				t.Errorf("missing map keys in nested map")
			}
			if err != nil {
				t.Errorf("unexpected error in map: %v", err)
			}

			if len(testCase.expected) != len(gates) {
				t.Errorf("expected feature gate size %d, got %d", len(testCase.expected), len(gates))
			}

			for k, v := range testCase.expected {
				gateVal, ok := gates[k]
				if !ok {
					t.Errorf("featureGates missing key %q", k)
					continue
				}

				if v != gateVal {
					t.Errorf("expected value %v, got %v", v, gateVal)
				}
			}
		})
	}
}
