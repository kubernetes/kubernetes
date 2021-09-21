/*
Copyright 2021 The Kubernetes Authors.

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

package tracing

import (
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/apiserver"
)

var (
	localhost    = "localhost:4317"
	ipAddress    = "127.0.0.1:4317"
	samplingRate = int32(12345)
)

func strptr(s string) *string {
	return &s
}

func TestReadTracingConfiguration(t *testing.T) {
	testcases := []struct {
		name           string
		contents       string
		createFile     bool
		expectedResult *apiserver.TracingConfiguration
		expectedError  *string
	}{
		{
			name:           "empty",
			createFile:     true,
			contents:       ``,
			expectedResult: &apiserver.TracingConfiguration{},
			expectedError:  nil,
		},
		{
			name:           "absent",
			createFile:     false,
			contents:       ``,
			expectedResult: nil,
			expectedError:  strptr("unable to read tracing configuration from \"test-tracing-config-absent\": open test-tracing-config-absent: no such file or directory"),
		},
		{
			name:       "v1alpha1",
			createFile: true,
			contents: `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: TracingConfiguration
endpoint: localhost:4317
samplingRatePerMillion: 12345
`,
			expectedResult: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				Endpoint:               &localhost,
				SamplingRatePerMillion: &samplingRate,
			},
			expectedError: nil,
		},
		{
			name:       "ip address",
			createFile: true,
			contents: `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: TracingConfiguration
endpoint: 127.0.0.1:4317
`,
			expectedResult: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				Endpoint: &ipAddress,
			},
			expectedError: nil,
		},
		{
			name:       "wrong_type",
			createFile: true,
			contents: `
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: agent
spec:
  selector:
    matchLabels:
      k8s-app: agent
  template:
    metadata:
      labels:
        k8s-app: agent
    spec:
      containers:
        - image: k8s.gcr.io/busybox
          name: agent
`,
			expectedResult: nil,
			expectedError:  strptr("unable to decode tracing configuration data: no kind \"DaemonSet\" is registered for version \"apps/v1\" in scheme"),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			proxyConfig := fmt.Sprintf("test-tracing-config-%s", tc.name)
			if tc.createFile {
				f, err := ioutil.TempFile("", proxyConfig)
				if err != nil {
					t.Fatal(err)
				}
				defer os.Remove(f.Name())
				if err := ioutil.WriteFile(f.Name(), []byte(tc.contents), os.FileMode(0755)); err != nil {
					t.Fatal(err)
				}
				proxyConfig = f.Name()
			}
			config, err := ReadTracingConfiguration(proxyConfig)
			if err == nil && tc.expectedError != nil {
				t.Errorf("calling ReadTracingConfiguration expected error: %s, did not get it", *tc.expectedError)
			}
			if err != nil && tc.expectedError == nil {
				t.Errorf("unexpected error calling ReadTracingConfiguration got: %#v", err)
			}
			if err != nil && tc.expectedError != nil && !strings.HasPrefix(err.Error(), *tc.expectedError) {
				t.Errorf("calling ReadTracingConfiguration expected error: %s, got %#v", *tc.expectedError, err)
			}
			if !reflect.DeepEqual(config, tc.expectedResult) {
				t.Errorf("problem with configuration returned from ReadTracingConfiguration expected: %#v, got: %#v", tc.expectedResult, config)
			}
		})
	}
}

func TestValidateTracingConfiguration(t *testing.T) {
	samplingRate := int32(12378)
	negativeRate := int32(-1)
	tooHighRate := int32(1000001)
	validEndpoint := "localhost:4317"
	dnsEndpoint := "dns://google.com:4317"
	unixEndpoint := "unix://path/to/socket"
	invalidURL := "dn%2s://localhost:4317"
	httpEndpoint := "http://localhost:4317"
	testcases := []struct {
		name        string
		expectError bool
		contents    *apiserver.TracingConfiguration
	}{
		{
			name:        "sampling-rate-valid",
			expectError: false,
			contents: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				SamplingRatePerMillion: &samplingRate,
			},
		},
		{
			name:        "sampling-rate-negative",
			expectError: true,
			contents: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				SamplingRatePerMillion: &negativeRate,
			},
		},
		{
			name:        "sampling-rate-negative",
			expectError: true,
			contents: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				SamplingRatePerMillion: &tooHighRate,
			},
		},
		{
			name:        "default Endpoint",
			expectError: false,
			contents: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				Endpoint: &validEndpoint,
			},
		},
		{
			name:        "dns Endpoint",
			expectError: false,
			contents: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				Endpoint: &dnsEndpoint,
			},
		},
		{
			name:        "unix Endpoint",
			expectError: false,
			contents: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				Endpoint: &unixEndpoint,
			},
		},
		{
			name:        "invalid Endpoint",
			expectError: true,
			contents: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				Endpoint: &httpEndpoint,
			},
		},
		{
			name:        "invalid url",
			expectError: true,
			contents: &apiserver.TracingConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				Endpoint: &invalidURL,
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			errs := ValidateTracingConfiguration(tc.contents)
			if tc.expectError == false && len(errs) != 0 {
				t.Errorf("Calling ValidateTracingConfiguration expected no error, got %v", errs)
			} else if tc.expectError == true && len(errs) == 0 {
				t.Errorf("Calling ValidateTracingConfiguration expected error, got no error")
			}
		})
	}
}
