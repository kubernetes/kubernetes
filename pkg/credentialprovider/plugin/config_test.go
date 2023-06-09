/*
Copyright 2020 The Kubernetes Authors.

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

package plugin

import (
	"os"
	"reflect"
	"testing"
	"time"

	utiltesting "k8s.io/client-go/util/testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
)

func Test_readCredentialProviderConfigFile(t *testing.T) {
	testcases := []struct {
		name       string
		configData string
		config     *kubeletconfig.CredentialProviderConfig
		expectErr  bool
	}{
		{
			name: "config with 1 plugin and 1 image matcher",
			configData: `---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1alpha1
providers:
  - name: test
    matchImages:
    - "registry.io/foobar"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`,
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "test",
						MatchImages:          []string{"registry.io/foobar"},
						DefaultCacheDuration: &metav1.Duration{Duration: 10 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
						Args:                 []string{"--v=5"},
						Env: []kubeletconfig.ExecEnvVar{
							{
								Name:  "FOO",
								Value: "BAR",
							},
						},
					},
				},
			},
		},
		{
			name: "config with 1 plugin and a wildcard image match",
			configData: `---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1alpha1
providers:
  - name: test
    matchImages:
    - "registry.io/*"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`,
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "test",
						MatchImages:          []string{"registry.io/*"},
						DefaultCacheDuration: &metav1.Duration{Duration: 10 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
						Args:                 []string{"--v=5"},
						Env: []kubeletconfig.ExecEnvVar{
							{
								Name:  "FOO",
								Value: "BAR",
							},
						},
					},
				},
			},
		},
		{
			name: "config with 1 plugin and multiple image matchers",
			configData: `---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1alpha1
providers:
  - name: test
    matchImages:
    - "registry.io/*"
    - "foobar.registry.io/*"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`,
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "test",
						MatchImages:          []string{"registry.io/*", "foobar.registry.io/*"},
						DefaultCacheDuration: &metav1.Duration{Duration: 10 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
						Args:                 []string{"--v=5"},
						Env: []kubeletconfig.ExecEnvVar{
							{
								Name:  "FOO",
								Value: "BAR",
							},
						},
					},
				},
			},
		},
		{
			name: "config with multiple providers",
			configData: `---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1alpha1
providers:
  - name: test1
    matchImages:
    - "registry.io/one"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
  - name: test2
    matchImages:
    - "registry.io/two"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`,

			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "test1",
						MatchImages:          []string{"registry.io/one"},
						DefaultCacheDuration: &metav1.Duration{Duration: 10 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
					{
						Name:                 "test2",
						MatchImages:          []string{"registry.io/two"},
						DefaultCacheDuration: &metav1.Duration{Duration: 10 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
						Args:                 []string{"--v=5"},
						Env: []kubeletconfig.ExecEnvVar{
							{
								Name:  "FOO",
								Value: "BAR",
							},
						},
					},
				},
			},
		},
		{
			name: "v1beta1 config with multiple providers",
			configData: `---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1beta1
providers:
  - name: test1
    matchImages:
    - "registry.io/one"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1beta1
  - name: test2
    matchImages:
    - "registry.io/two"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1beta1
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`,

			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "test1",
						MatchImages:          []string{"registry.io/one"},
						DefaultCacheDuration: &metav1.Duration{Duration: 10 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1beta1",
					},
					{
						Name:                 "test2",
						MatchImages:          []string{"registry.io/two"},
						DefaultCacheDuration: &metav1.Duration{Duration: 10 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1beta1",
						Args:                 []string{"--v=5"},
						Env: []kubeletconfig.ExecEnvVar{
							{
								Name:  "FOO",
								Value: "BAR",
							},
						},
					},
				},
			},
		},
		{
			name: "v1 config with multiple providers",
			configData: `---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: test1
    matchImages:
    - "registry.io/one"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1
  - name: test2
    matchImages:
    - "registry.io/two"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`,

			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "test1",
						MatchImages:          []string{"registry.io/one"},
						DefaultCacheDuration: &metav1.Duration{Duration: 10 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
					},
					{
						Name:                 "test2",
						MatchImages:          []string{"registry.io/two"},
						DefaultCacheDuration: &metav1.Duration{Duration: 10 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						Args:                 []string{"--v=5"},
						Env: []kubeletconfig.ExecEnvVar{
							{
								Name:  "FOO",
								Value: "BAR",
							},
						},
					},
				},
			},
		},
		{
			name: "config with wrong Kind",
			configData: `---
kind: WrongKind
apiVersion: kubelet.config.k8s.io/v1alpha1
providers:
  - name: test
    matchImages:
    - "registry.io/foobar"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`,
			config:    nil,
			expectErr: true,
		},
		{
			name: "config with wrong apiversion",
			configData: `---
kind: CredentialProviderConfig
apiVersion: foobar/v1alpha1
providers:
  - name: test
    matchImages:
    - "registry.io/foobar"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`,
			config:    nil,
			expectErr: true,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			file, err := os.CreateTemp("", "config.yaml")
			if err != nil {
				t.Fatal(err)
			}
			defer utiltesting.CloseAndRemove(t, file)

			_, err = file.WriteString(testcase.configData)
			if err != nil {
				t.Fatal(err)
			}

			authConfig, err := readCredentialProviderConfigFile(file.Name())
			if err != nil && !testcase.expectErr {
				t.Fatal(err)
			}

			if err == nil && testcase.expectErr {
				t.Error("expected error but got none")
			}

			if !reflect.DeepEqual(authConfig, testcase.config) {
				t.Logf("actual auth config: %#v", authConfig)
				t.Logf("expected auth config: %#v", testcase.config)
				t.Error("credential provider config did not match")
			}
		})
	}
}

func Test_validateCredentialProviderConfig(t *testing.T) {
	testcases := []struct {
		name      string
		config    *kubeletconfig.CredentialProviderConfig
		shouldErr bool
	}{
		{
			name:      "no providers provided",
			config:    &kubeletconfig.CredentialProviderConfig{},
			shouldErr: true,
		},
		{
			name: "no matchImages provided",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "no default cache duration provided",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:        "foobar",
						MatchImages: []string{"foobar.registry.io"},
						APIVersion:  "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "name contains '/'",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foo/../bar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "name is '.'",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 ".",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "name is '..'",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "..",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "name contains spaces",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foo bar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "no apiVersion",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "invalid apiVersion",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha0",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "negative default cache duration",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: -1 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "invalid match image",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"%invalid%"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			shouldErr: true,
		},
		{
			name: "valid config",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			shouldErr: false,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			errs := validateCredentialProviderConfig(testcase.config)

			if testcase.shouldErr && len(errs) == 0 {
				t.Errorf("expected error but got none")
			} else if !testcase.shouldErr && len(errs) > 0 {
				t.Errorf("expected no error but received errors: %v", errs.ToAggregate())

			}
		})
	}
}
