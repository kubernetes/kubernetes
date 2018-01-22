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

package eventratelimit

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	eventratelimitapi "k8s.io/kubernetes/plugin/pkg/admission/eventratelimit/apis/eventratelimit"
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
		ExpectedConfig *eventratelimitapi.Configuration
	}{
		// valid configuration
		{
			CaseName: "valid configuration",
			ConfigBody: `
apiVersion: eventratelimit.admission.k8s.io/v1alpha1
kind: Configuration
limits:
  - type: Namespace
    qps: 50
    burst: 100
    cacheSize: 2000
  - type: User
    qps: 10
    burst: 50
`,
			ExpecteErr: false,
			ExpectedConfig: &eventratelimitapi.Configuration{
				Limits: []eventratelimitapi.Limit{
					{
						Type:      "Namespace",
						QPS:       50,
						Burst:     100,
						CacheSize: 2000,
					},
					{
						Type:  "User",
						QPS:   10,
						Burst: 50,
					},
				},
			},
		},

		// invalid configuration apiVersion
		{
			CaseName: "invalid configuration apiVersion",
			ConfigBody: `
apiVersion: eventratelimit.admission.k8s.io/v1alpha2
kind: Configuration
limits:
  - type: Namespace
    qps: 50
    burst: 100
    cacheSize: 2000
  - type: User
    qps: 10
    burst: 50
`,
			ExpecteErr: true,
		},

		// invalid configuration Limit
		{
			CaseName: "invalid configuration Limit",
			ConfigBody: `
apiVersion: eventratelimit.admission.k8s.io/v1alpha1
kind: Configuration
invalid-limits:
  - type: Namespace
    qps: 50
    burst: 100
    cacheSize: 2000
  - type: User
    qps: 10
    burst: 50
`,
			ExpecteErr:     false,
			ExpectedConfig: &eventratelimitapi.Configuration{},
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
				t.Errorf("[%s] expect error but got none", testcase.CaseName)
			}
			if !testcase.ExpecteErr && err != nil {
				t.Errorf("[%s] unexpected error: %v", testcase.CaseName, err)
			}
			if !reflect.DeepEqual(testcase.ExpectedConfig, config) {
				t.Errorf("[%s] expectedconfig: %v got %v", testcase.CaseName, testcase.ExpectedConfig, config)
			}
		}()
	}
}
