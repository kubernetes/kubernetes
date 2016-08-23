/*
Copyright 2016 The Kubernetes Authors.

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

package restclient

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/testapi"
)

func TestIsConfigTransportTLS(t *testing.T) {
	testCases := []struct {
		Config       *Config
		TransportTLS bool
		Err          bool
	}{
		{
			Config:       &Config{},
			TransportTLS: false,
		},
		{
			Config: &Config{
				Host: "https://localhost",
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host: "localhost",
				TLSClientConfig: TLSClientConfig{
					CertFile: "foo",
				},
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host: "///:://localhost",
				TLSClientConfig: TLSClientConfig{
					CertFile: "foo",
				},
			},
			TransportTLS: false,
		},
		{
			Config: &Config{
				Host:     "1.2.3.4:567",
				Insecure: true,
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host:           "https://localhost",
				AlternateHosts: []string{"https://10.10.0.2", "https://10.10.0.3"},
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				AlternateHosts: []string{"https://10.10.0.2", "https://10.10.0.3"},
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host:           "http://localhost",
				AlternateHosts: []string{"https://10.10.0.2", "https://10.10.0.3"},
			},
			Err: true,
		},
		{
			Config: &Config{
				Host:           "https://10.10.0.1",
				AlternateHosts: []string{"http://10.10.0.2", "https://10.10.0.3"},
			},
			Err: true,
		},
	}
	for i, testCase := range testCases {
		if err := SetKubernetesDefaults(testCase.Config); err != nil {
			t.Errorf("%d: setting defaults failed for %#v: %v", i, testCase.Config, err)
			continue
		}
		useTLS, err := IsConfigTransportTLS(*testCase.Config)
		isErr := err != nil
		if isErr != testCase.Err {
			t.Errorf("%d: Unexpected error %v", i, err)
		}
		if !isErr && testCase.TransportTLS != useTLS {
			t.Errorf("%d: expected %v for %#v", i, testCase.TransportTLS, testCase.Config)
		}
	}
}

func TestSetKubernetesDefaultsUserAgent(t *testing.T) {
	config := &Config{}
	if err := SetKubernetesDefaults(config); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !strings.Contains(config.UserAgent, "kubernetes/") {
		t.Errorf("no user agent set: %#v", config)
	}
}

func TestRESTClientRequires(t *testing.T) {
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", ContentConfig: ContentConfig{NegotiatedSerializer: testapi.Default.NegotiatedSerializer()}}); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", ContentConfig: ContentConfig{GroupVersion: testapi.Default.GroupVersion()}}); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := RESTClientFor(&Config{Host: "127.0.0.1", ContentConfig: ContentConfig{GroupVersion: testapi.Default.GroupVersion(), NegotiatedSerializer: testapi.Default.NegotiatedSerializer()}}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
