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

package token

import (
	"testing"
	"time"

	"github.com/pkg/errors"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestFetchKubeConfigWithTimeout(t *testing.T) {
	const testAPIEndpoint = "sample-endpoint:1234"
	tests := []struct {
		name             string
		discoveryTimeout time.Duration
		shouldFail       bool
	}{
		{
			name:             "Timeout if value is not returned on time",
			discoveryTimeout: 1 * time.Second,
			shouldFail:       true,
		},
		{
			name:             "Don't timeout if value is returned on time",
			discoveryTimeout: 5 * time.Second,
			shouldFail:       false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cfg, err := fetchKubeConfigWithTimeout(testAPIEndpoint, test.discoveryTimeout, func(apiEndpoint string) (*clientcmdapi.Config, error) {
				if apiEndpoint != testAPIEndpoint {
					return nil, errors.Errorf("unexpected API server endpoint:\n\texpected: %q\n\tgot: %q", testAPIEndpoint, apiEndpoint)
				}

				time.Sleep(3 * time.Second)
				return &clientcmdapi.Config{}, nil
			})

			if test.shouldFail {
				if err == nil {
					t.Fatal("unexpected success")
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected failure: %v", err)
				}
				if cfg == nil {
					t.Fatal("cfg is nil")
				}
			}
		})
	}
}
