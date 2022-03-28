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

package genericclioptions

import (
	"fmt"
	"io/ioutil"
	"k8s.io/client-go/rest"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"os"
	"testing"
)

func TestConfigFlagsClientsInitialization(t *testing.T) {
	testCases := []struct {
		name                string
		usePersistentConfig bool
	}{
		{
			name:                "persists clients (returns same clients) and reacts to SetStdinInUse",
			usePersistentConfig: true,
		},
		{
			name:                "does not persists clients (returns different clients) and reacts to SetStdinInUse",
			usePersistentConfig: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// create config for testing
			tmpFile, err := ioutil.TempFile(os.TempDir(), "httpclienttest_temp")
			if err != nil {
				t.Fatalf(fmt.Sprintf("unable to create a fake client config: %v", err))
			}
			kubeConfig := tmpFile.Name()
			defer os.Remove(kubeConfig)

			flags := NewConfigFlags(tc.usePersistentConfig).WithWrapConfigFn(func(config *rest.Config) *rest.Config {
				// simulate Exec Config Provider
				config.ExecProvider = &clientcmdapi.ExecConfig{
					Command:    "cmd",
					Args:       []string{"get-credentials"},
					APIVersion: "client.authentication.k8s.io/v1",
				}
				return config
			})
			flags.KubeConfig = &kubeConfig

			// test SetStdinInUse not set in the config
			config, err := flags.ToRESTConfig()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if config.ExecProvider.StdinUnavailable || len(config.ExecProvider.StdinUnavailableMessage) != 0 {
				t.Fatalf("ExecProvider StdinUnavailable should not be initialized: %#v", config.ExecProvider)
			}

			// test clients are/are not getting cached
			httpClient, err := flags.ToHTTPClient()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			httpClient2, err := flags.ToHTTPClient()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			discoveryClient, err := flags.ToDiscoveryClient()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			discoveryClient2, err := flags.ToDiscoveryClient()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if httpClient == httpClient2 && discoveryClient == discoveryClient2 {
				if !tc.usePersistentConfig {
					t.Fatalf("clients should differ")
				}
			} else {
				if tc.usePersistentConfig {
					t.Fatalf("clients should not differ")
				}
			}

			// test SetStdinInUse set properly in the config
			flags.SetStdinInUse()
			config, err = flags.ToRESTConfig()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !config.ExecProvider.StdinUnavailable || len(config.ExecProvider.StdinUnavailableMessage) == 0 {
				t.Fatalf("ExecProvider StdinUnavailable was not properly initialized: %#v", config.ExecProvider)
			}

			// test invalidation of clients
			httpClient3, err := flags.ToHTTPClient()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			discoveryClient3, err := flags.ToDiscoveryClient()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if httpClient2 == httpClient3 || discoveryClient2 == discoveryClient3 {
				t.Fatalf("caching of clients should be invalidated to apply change in stdin")
			}

			// test SetStdinInUse idempotence
			flags.SetStdinInUse()
			httpClient4, err := flags.ToHTTPClient()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			discoveryClient4, err := flags.ToDiscoveryClient()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if httpClient3 == httpClient4 && discoveryClient3 == discoveryClient4 {
				if !tc.usePersistentConfig {
					t.Fatalf("clients should differ")
				}
			} else {
				if tc.usePersistentConfig {
					t.Fatalf("multiple invocations of SetStdinInUse should not invalidate the clients")
				}
			}
		})
	}
}
