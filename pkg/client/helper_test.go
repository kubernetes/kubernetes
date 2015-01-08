/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

import (
	"net/http"
	"testing"
)

func TestTransportFor(t *testing.T) {
	testCases := map[string]struct {
		Config  *Config
		Err     bool
		Default bool
	}{
		"default transport": {
			Config: &Config{},
		},
	}
	for k, testCase := range testCases {
		transport, err := TransportFor(testCase.Config)
		switch {
		case testCase.Err && err == nil:
			t.Errorf("%s: unexpected non-error", k)
			continue
		case !testCase.Err && err != nil:
			t.Errorf("%s: unexpected error: %v", k, err)
			continue
		}
		if testCase.Default && transport != http.DefaultTransport {
			t.Errorf("%s: expected the default transport, got %#v", k, transport)
		}
	}
}

func TestIsConfigTransportTLS(t *testing.T) {
	testCases := []struct {
		Config       *Config
		TransportTLS bool
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
				Host:     "localhost",
				CertFile: "foo",
			},
			TransportTLS: true,
		},
		{
			Config: &Config{
				Host:     "///:://localhost",
				CertFile: "foo",
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
	}
	for _, testCase := range testCases {
		if err := SetKubernetesDefaults(testCase.Config); err != nil {
			t.Errorf("setting defaults failed for %#v: %v", testCase.Config, err)
			continue
		}
		useTLS := IsConfigTransportTLS(*testCase.Config)
		if testCase.TransportTLS != useTLS {
			t.Errorf("expected %v for %#v", testCase.TransportTLS, testCase.Config)
		}
	}
}
