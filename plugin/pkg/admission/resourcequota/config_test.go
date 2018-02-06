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

package resourcequota

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	resourcequotaapi "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota"
)

func TestLoadConfiguration(t *testing.T) {
	// create a place holder file to hold per test config
	configFile, err := ioutil.TempFile("", "admission-config")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	configFileName := configFile.Name()
	defer os.Remove(configFileName)

	if err = configFile.Close(); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	// individual test scenarios
	testCases := []struct {
		CaseName       string
		ConfigBody     string
		ExpecteErr     bool
		ExpectedConfig *resourcequotaapi.Configuration
	}{
		// valid configuration
		{
			CaseName: "valid configuration",
			ConfigBody: `
apiVersion: resourcequota.admission.k8s.io/v1alpha1
kind: Configuration
metadata:
  name: quota-config
limitedResources:
  - resource: persistentvolumeclaims
    matchContains:
    - requests.storage
  - resource: services
    matchContains:
      - loadbalancers
      - nodeports
  - resource: pods
    matchContains:
      - pods
      - cpu
`,
			ExpecteErr: false,
			ExpectedConfig: &resourcequotaapi.Configuration{
				LimitedResources: []resourcequotaapi.LimitedResource{
					{
						Resource:      "persistentvolumeclaims",
						MatchContains: []string{"requests.storage"},
					},
					{
						Resource:      "services",
						MatchContains: []string{"loadbalancers", "nodeports"},
					},
					{
						Resource:      "pods",
						MatchContains: []string{"pods", "cpu"},
					},
				},
			},
		},

		// invalid configuration kind
		{
			CaseName: "invalid configuration kind",
			ConfigBody: `
apiVersion: resourcequota.admission.k8s.io/v1alpha1
kind: invalid-Configuration
metadata:
  name: quota-config
limitedResources:
  - resource: persistentvolumeclaims
    matchContains:
    - requests.storage
  - resource: services
    matchContains:
      - loadbalancers
      - nodeports
  - resource: pods
    matchContains:
      - pods
      - cpu
`,
			ExpecteErr:     true,
			ExpectedConfig: nil,
		},

		// invalid configuration
		{
			CaseName: "invalid configuration",
			ConfigBody: `
apiVersion: resourcequota.admission.k8s.io/v1alpha1
kind: Configuration
metadata:
  name: quota-config
invalid-limitedResources:
  - resource: persistentvolumeclaims
    matchContains:
    - requests.storage
`,
			ExpecteErr:     false,
			ExpectedConfig: &resourcequotaapi.Configuration{},
		},
	}

	for _, testcase := range testCases {
		func() {
			if err = ioutil.WriteFile(configFileName, []byte(testcase.ConfigBody), 0644); err != nil {
				t.Fatalf("unexpected err writing temp file: %v", err)
			}

			configFile, err := os.Open(configFileName)
			if err != nil {
				t.Fatalf("failed to read test config: %v", err)
			}
			defer configFile.Close()

			config, err := LoadConfiguration(configFile)
			if testcase.ExpecteErr && err == nil {
				t.Errorf("expect error but got none")
			}
			if !testcase.ExpecteErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if !reflect.DeepEqual(testcase.ExpectedConfig, config) {
				t.Errorf("[%s] expectedconfig: %v got %v", testcase.CaseName, testcase.ExpectedConfig, config)
			}
		}()
	}
}
