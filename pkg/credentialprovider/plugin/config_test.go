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
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	utiltesting "k8s.io/client-go/util/testing"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/utils/ptr"
)

func Test_readCredentialProviderConfig(t *testing.T) {
	testcases := []struct {
		name               string
		configData         []string // Array to support multiple files for directory tests
		fileNames          []string // Optional file names for directory tests
		isDir              bool     // Whether to create a directory with multiple files
		config             *kubeletconfig.CredentialProviderConfig
		expectErr          string
		expectedConfigHash string // Expected hash of the config
	}{
		{
			name:       "empty directory with no JSON or YAML files",
			configData: []string{},
			isDir:      true,
			config:     nil,
			expectErr:  "no configuration files found in directory",
		},
		{
			name: "directory with unsupported file extensions",
			configData: []string{
				`This is a text file with unsupported extension`,
				`This is another text file with unsupported extension`,
			},
			fileNames: []string{
				"config.txt",
				"config.md",
			},
			isDir:     true,
			config:    nil,
			expectErr: "no configuration files found in directory",
		},
		{
			name: "config with 1 plugin and 1 image matcher",
			configData: []string{`---
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
      value: BAR`},
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
			expectedConfigHash: "sha256:8a62755289ba50c4ca9495baab69eb861068503f8bd49b853e8ba6cf95c72bb8",
		},
		{
			name: "config with 1 plugin and 1 image matcher (JSON!)",
			configData: []string{`{
				  "kind": "CredentialProviderConfig",
				  "apiVersion": "kubelet.config.k8s.io/v1alpha1",
				  "providers": [
					{
					  "name": "test",
					  "matchImages": [
						"registry.io/foobar"
					  ],
					  "defaultCacheDuration": "10m",
					  "apiVersion": "credentialprovider.kubelet.k8s.io/v1alpha1",
					  "args": [
						"--v=5"
					  ],
					  "env": [
						{
						  "name": "FOO",
						  "value": "BAR"
						}
					  ]
					}
				  ]
				}`,
			},
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
			expectedConfigHash: "sha256:fd0946c206cc5b8735cd57816f21e139cfe480f27119f1d1f80e7c2fd9dc4636",
		},
		{
			name: "config with 1 plugin and a wildcard image match",
			configData: []string{`---
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
      value: BAR`},
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
			expectedConfigHash: "sha256:5272b0ed7da9c85912217bb9e5293549d7a03cdd510cb73df8d06bd8db363921",
		},
		{
			name: "config with 1 plugin and multiple image matchers",
			configData: []string{`---
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
      value: BAR`},
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
			expectedConfigHash: "sha256:e167993fa14ef8a799aebe1ebca7177478f93b8ada2b61286479641a4266d75e",
		},
		{
			name: "config with multiple providers",
			configData: []string{`---
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
      value: BAR`},

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
			expectedConfigHash: "sha256:04d747035e475d4fed4f2e2ec061941d401172e4447e2262a6269f22a670418b",
		},
		{
			name: "v1beta1 config with multiple providers",
			configData: []string{`---
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
      value: BAR`},

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
			expectedConfigHash: "sha256:5d6e9671b548ddcaf4674cfa5e13257bcff06c931a8a4ef13f2577f94ff4cff3",
		},
		{
			name: "v1 config with multiple providers",
			configData: []string{`---
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
      value: BAR`},

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
			expectedConfigHash: "sha256:ea93e932f6b7f0ab45cecd6e7141cf5d3a6a037868b59725a60e99a2b702e2a7",
		},
		{
			name: "config with wrong Kind",
			configData: []string{`---
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
      value: BAR`},
			config:    nil,
			expectErr: `no kind "WrongKind" is registered for version "kubelet.config.k8s.io/v1alpha1"`,
		},
		{
			name: "config with wrong apiversion",
			configData: []string{`---
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
      value: BAR`},
			config:    nil,
			expectErr: `no kind "CredentialProviderConfig" is registered for version "foobar/v1alpha1`,
		},
		{
			name: "config with invalid typo",
			configData: []string{`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: test
    matchImages:
    - "registry.io/foobar"
    defaultCacheDuration: 10m
    unknownField: should not be here # this field should not be here
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`},
			config:    nil,
			expectErr: `strict decoding error: unknown field "providers[0].unknownField"`,
		},
		{
			name: "v1alpha1 config with token attributes should fail",
			configData: []string{`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1alpha1
providers:
  - name: test
    matchImages:
    - "registry.io/foobar"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
    tokenAttributes:
      serviceAccountTokenAudience: audience
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`},
			config:    nil,
			expectErr: `strict decoding error: unknown field "providers[0].tokenAttributes"`,
		},
		{
			name: "v1beta1 config with token attributes should fail",
			configData: []string{`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1beta1
providers:
  - name: test
    matchImages:
    - "registry.io/foobar"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1beta1
    tokenAttributes:
      serviceAccountTokenAudience: audience
    args:
    - --v=5
    env:
    - name: FOO
      value: BAR`},
			config:    nil,
			expectErr: `strict decoding error: unknown field "providers[0].tokenAttributes"`,
		},
		{
			name: "directory with multiple config files in lexicographic order",
			configData: []string{
				`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: test1
    matchImages:
    - "registry.io/one"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1`,
				`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: test2
    matchImages:
    - "registry.io/two"
    defaultCacheDuration: 5m
    apiVersion: credentialprovider.kubelet.k8s.io/v1`,
			},
			isDir: true,
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
						DefaultCacheDuration: &metav1.Duration{Duration: 5 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
					},
				},
			},
			expectedConfigHash: "sha256:efef09979a19eee4802fe4b7aa46d5122f16687a8419443647182c1920f76ac9",
		},
		{
			name: "directory with mixed API versions in config files",
			configData: []string{
				`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1beta1
providers:
  - name: test-beta
    matchImages:
    - "beta.registry.io/*"
    defaultCacheDuration: 15m
    apiVersion: credentialprovider.kubelet.k8s.io/v1beta1`,
				`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: test-v1
    matchImages:
    - "v1.registry.io/*"
    defaultCacheDuration: 20m
    apiVersion: credentialprovider.kubelet.k8s.io/v1`,
				`{
  "kind": "CredentialProviderConfig",
  "apiVersion": "kubelet.config.k8s.io/v1",
  "providers": [
    {
      "name": "test-v2",
      "matchImages": [
        "v2.registry.io/*"
      ],
      "defaultCacheDuration": "20m",
      "apiVersion": "credentialprovider.kubelet.k8s.io/v1"
    }
  ]
}`,
			},
			isDir: true,
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "test-beta",
						MatchImages:          []string{"beta.registry.io/*"},
						DefaultCacheDuration: &metav1.Duration{Duration: 15 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1beta1",
					},
					{
						Name:                 "test-v1",
						MatchImages:          []string{"v1.registry.io/*"},
						DefaultCacheDuration: &metav1.Duration{Duration: 20 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
					},
					{
						Name:                 "test-v2",
						MatchImages:          []string{"v2.registry.io/*"},
						DefaultCacheDuration: &metav1.Duration{Duration: 20 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
					},
				},
			},
			expectedConfigHash: "sha256:292a674763a5b34f4e51bd0d4c5736375f31a04cb1bed519ca3fcd033ce3e20e",
		},
		{
			name: "directory with duplicate provider names, throw error",
			configData: []string{
				`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: duplicate
    matchImages:
    - "registry.io/one"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1`,
				`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: duplicate
    matchImages:
    - "registry.io/two"
    defaultCacheDuration: 5m
    apiVersion: credentialprovider.kubelet.k8s.io/v1`,
			},
			isDir:     true,
			config:    nil,
			expectErr: `duplicate provider name "duplicate" found in configuration file(s)`,
		},
		{
			name: "directory with mixed supported and unsupported file extensions",
			configData: []string{
				`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: test1
    matchImages:
    - "registry.io/one"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1`,
				`This is a text file with unsupported extension that should be skipped`,
				`{
  "kind": "CredentialProviderConfig",
  "apiVersion": "kubelet.config.k8s.io/v1",
  "providers": [
    {
      "name": "test2",
      "matchImages": [
        "registry.io/two"
      ],
      "defaultCacheDuration": "5m",
      "apiVersion": "credentialprovider.kubelet.k8s.io/v1"
    }
  ]
}`,
			},
			fileNames: []string{
				"config-001.yaml",
				"config-002.txt",
				"config-003.json",
			},
			isDir: true,
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
						DefaultCacheDuration: &metav1.Duration{Duration: 5 * time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
					},
				},
			},
			expectedConfigHash: "sha256:70b4a5afe55ad1045c14f427e33ef10d1bbd82fb0c5b6c944e2c9f8ad2b5d180",
		},
		{
			name: "directory with one invalid config file",
			configData: []string{
				`---
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: test1
    matchImages:
    - "registry.io/one"
    defaultCacheDuration: 10m
    apiVersion: credentialprovider.kubelet.k8s.io/v1`,
				`---
kind: WrongKind
apiVersion: kubelet.config.k8s.io/v1
providers:
  - name: test2
    matchImages:
    - "registry.io/two"
    defaultCacheDuration: 5m
    apiVersion: credentialprovider.kubelet.k8s.io/v1`,
			},
			isDir:     true,
			config:    nil,
			expectErr: `no kind "WrongKind" is registered for version "kubelet.config.k8s.io/v1"`,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			var configPath string
			var err error

			if testcase.isDir {
				// Create a temporary directory for multiple config files
				tempDir, err := os.MkdirTemp("", "config-dir")
				if err != nil {
					t.Fatal(err)
				}
				defer func() {
					if err := os.RemoveAll(tempDir); err != nil {
						t.Errorf("failed to remove temp directory: %v", err)
					}
				}()

				for i, configData := range testcase.configData {
					var fileName string
					if testcase.fileNames != nil && i < len(testcase.fileNames) {
						fileName = testcase.fileNames[i]
					} else {
						fileName = fmt.Sprintf("config-%03d.yaml", i)
					}
					filePath := fmt.Sprintf("%s/%s", tempDir, fileName)

					file, err := os.Create(filePath)
					if err != nil {
						t.Fatal(err)
					}

					if _, err = file.WriteString(configData); err != nil {
						if cerr := file.Close(); cerr != nil {
							t.Errorf("failed to close file: %v", cerr)
						}
						t.Fatal(err)
					}
					if err := file.Close(); err != nil {
						t.Errorf("failed to close file: %v", err)
					}
				}

				configPath = tempDir
			} else {
				// Create a single temporary file
				file, err := os.CreateTemp("", "config.yaml")
				if err != nil {
					t.Fatal(err)
				}
				defer utiltesting.CloseAndRemove(t, file)

				if len(testcase.configData) > 0 {
					if _, err = file.WriteString(testcase.configData[0]); err != nil {
						t.Fatal(err)
					}
				}

				configPath = file.Name()
			}

			authConfig, configHash, err := readCredentialProviderConfig(configPath)
			if len(testcase.expectErr) == 0 {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				if err == nil {
					t.Fatalf("expected error %q but got none", testcase.expectErr)
				}
				if !strings.Contains(err.Error(), testcase.expectErr) {
					t.Fatalf("expected error %q but got %q", testcase.expectErr, err.Error())
				}
			}

			if configHash != testcase.expectedConfigHash {
				t.Fatalf("expected config hash %q, got %q", testcase.expectedConfigHash, configHash)
			}

			if !reflect.DeepEqual(authConfig, testcase.config) {
				t.Fatalf("expected auth config: %v, got: %v", testcase.config, authConfig)
			}
		})
	}
}

func Test_validateCredentialProviderConfig(t *testing.T) {
	testcases := []struct {
		name                          string
		config                        *kubeletconfig.CredentialProviderConfig
		saTokenForCredentialProviders bool
		expectErr                     string
	}{
		{
			name:      "no providers provided",
			config:    &kubeletconfig.CredentialProviderConfig{},
			expectErr: `providers: Required value: at least 1 item in plugins is required`,
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
			expectErr: `providers.matchImages: Required value: at least 1 item in matchImages is required`,
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
			expectErr: `providers.defaultCacheDuration: Required value`,
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
			expectErr: `providers.name: Invalid value: "foo/../bar": provider name cannot contain '/'`,
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
			expectErr: `providers.name: Invalid value: ".": provider name cannot be '.'`,
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
			expectErr: `providers.name: Invalid value: "..": provider name cannot be '..'`,
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
			expectErr: `providers.name: Invalid value: "foo bar": provider name cannot contain spaces`,
		},
		{
			name: "duplicate names",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
					{
						Name:                 "foobar",
						MatchImages:          []string{"bar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
					},
				},
			},
			expectErr: `providers.name: Duplicate value: "foobar"`,
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
			expectErr: "providers.apiVersion: Required value",
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
			expectErr: `providers.apiVersion: Unsupported value: "credentialprovider.kubelet.k8s.io/v1alpha0": supported values: "credentialprovider.kubelet.k8s.io/v1", "credentialprovider.kubelet.k8s.io/v1alpha1", "credentialprovider.kubelet.k8s.io/v1beta1"`,
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
			expectErr: "providers.defaultCacheDuration: Invalid value: \"-1m0s\": must be greater than or equal to 0",
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
			expectErr: `providers.matchImages: Invalid value: "%invalid%": match image is invalid: parse "https://%invalid%": invalid URL escape "%in"`,
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
		},
		{
			name: "token attributes set without KubeletServiceAccountTokenForCredentialProviders feature gate enabled",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience: "audience",
							CacheType:                   kubeletconfig.ServiceAccountServiceAccountTokenCacheType,
							RequireServiceAccount:       ptr.To(true),
						},
					},
				},
			},
			expectErr: `providers.tokenAttributes: Forbidden: tokenAttributes is not supported when KubeletServiceAccountTokenForCredentialProviders feature gate is disabled`,
		},
		{
			name: "token attributes not nil but empty ServiceAccountTokenAudience",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							CacheType:                            kubeletconfig.TokenServiceAccountTokenCacheType,
							RequiredServiceAccountAnnotationKeys: []string{"prefix.io/annotation-1", "prefix.io/annotation-2"},
							RequireServiceAccount:                ptr.To(true),
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes.serviceAccountTokenAudience: Required value`,
		},
		{
			name: "token attributes not nil but empty CacheType",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience: "audience",
							RequireServiceAccount:       ptr.To(true),
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes.cacheType: Required value: cacheType is required to be set when tokenAttributes is specified. Supported values are: ServiceAccount, Token`,
		},
		{
			name: "token attributes not nil, invalid CacheType",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience: "audience",
							CacheType:                   "invalid-cache-type",
							RequireServiceAccount:       ptr.To(true),
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes.cacheType: Unsupported value: "invalid-cache-type": supported values: "ServiceAccount", "Token"`,
		},
		{
			name: "token attributes not nil but empty ServiceAccountTokenRequired",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.ServiceAccountServiceAccountTokenCacheType,
							RequiredServiceAccountAnnotationKeys: []string{"prefix.io/annotation-1", "prefix.io/annotation-2"},
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes.requireServiceAccount: Required value`,
		},
		{
			name: "required service account annotation keys not qualified name (same validation as metav1.ObjectMeta)",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.TokenServiceAccountTokenCacheType,
							RequireServiceAccount:                ptr.To(true),
							RequiredServiceAccountAnnotationKeys: []string{"cantendwithadash-", "now-with-dashes/simple"}, // first key is invalid
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes.requiredServiceAccountAnnotationKeys: Invalid value: "cantendwithadash-": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`,
		},
		{
			name: "optional service account annotation keys not qualified name (same validation as metav1.ObjectMeta)",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.ServiceAccountServiceAccountTokenCacheType,
							RequireServiceAccount:                ptr.To(true),
							OptionalServiceAccountAnnotationKeys: []string{"cantendwithadash-", "now-with-dashes/simple"}, // first key is invalid
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes.optionalServiceAccountAnnotationKeys: Invalid value: "cantendwithadash-": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`,
		},
		{
			name: "duplicate required service account annotation keys",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.TokenServiceAccountTokenCacheType,
							RequireServiceAccount:                ptr.To(true),
							RequiredServiceAccountAnnotationKeys: []string{"now-with-dashes/simple", "now-with-dashes/simple"},
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes.requiredServiceAccountAnnotationKeys: Duplicate value: "now-with-dashes/simple"`,
		},
		{
			name: "duplicate optional service account annotation keys",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.ServiceAccountServiceAccountTokenCacheType,
							RequireServiceAccount:                ptr.To(true),
							OptionalServiceAccountAnnotationKeys: []string{"now-with-dashes/simple", "now-with-dashes/simple"},
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes.optionalServiceAccountAnnotationKeys: Duplicate value: "now-with-dashes/simple"`,
		},
		{
			name: "annotation key in required and optional keys",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.TokenServiceAccountTokenCacheType,
							RequireServiceAccount:                ptr.To(true),
							RequiredServiceAccountAnnotationKeys: []string{"now-with-dashes/simple-1", "now-with-dashes/simple-2"},
							OptionalServiceAccountAnnotationKeys: []string{"now-with-dashes/simple-2", "now-with-dashes/simple-3"},
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes: Invalid value: ["now-with-dashes/simple-2"]: annotation keys cannot be both required and optional`,
		},
		{
			name: "required annotation keys set when requireServiceAccount is false",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.ServiceAccountServiceAccountTokenCacheType,
							RequireServiceAccount:                ptr.To(false),
							RequiredServiceAccountAnnotationKeys: []string{"now-with-dashes/simple-1", "now-with-dashes/simple-2"},
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes.requiredServiceAccountAnnotationKeys: Forbidden: requireServiceAccount cannot be false when requiredServiceAccountAnnotationKeys is set`,
		},
		{
			name: "valid config with KubeletServiceAccountTokenForCredentialProviders feature gate enabled",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.TokenServiceAccountTokenCacheType,
							RequireServiceAccount:                ptr.To(true),
							RequiredServiceAccountAnnotationKeys: []string{"now-with-dashes/simple-1", "now-with-dashes/simple-2"},
							OptionalServiceAccountAnnotationKeys: []string{"now-with-dashes/simple-3"},
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
		},
		{
			name: "tokenAttributes set with credentialprovider.kubelet.k8s.io/v1alpha1 APIVersion",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1alpha1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.TokenServiceAccountTokenCacheType,
							RequireServiceAccount:                ptr.To(true),
							RequiredServiceAccountAnnotationKeys: []string{"now-with-dashes/simple"},
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes: Forbidden: tokenAttributes is only supported for credentialprovider.kubelet.k8s.io/v1 API version`,
		},
		{
			name: "tokenAttributes set with credentialprovider.kubelet.k8s.io/v1beta1 APIVersion",
			config: &kubeletconfig.CredentialProviderConfig{
				Providers: []kubeletconfig.CredentialProvider{
					{
						Name:                 "foobar",
						MatchImages:          []string{"foobar.registry.io"},
						DefaultCacheDuration: &metav1.Duration{Duration: time.Minute},
						APIVersion:           "credentialprovider.kubelet.k8s.io/v1beta1",
						TokenAttributes: &kubeletconfig.ServiceAccountTokenAttributes{
							ServiceAccountTokenAudience:          "audience",
							CacheType:                            kubeletconfig.ServiceAccountServiceAccountTokenCacheType,
							RequireServiceAccount:                ptr.To(true),
							RequiredServiceAccountAnnotationKeys: []string{"now-with-dashes/simple"},
						},
					},
				},
			},
			saTokenForCredentialProviders: true,
			expectErr:                     `providers.tokenAttributes: Forbidden: tokenAttributes is only supported for credentialprovider.kubelet.k8s.io/v1 API version`,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			errs := validateCredentialProviderConfig(testcase.config, testcase.saTokenForCredentialProviders).ToAggregate()
			if d := cmp.Diff(testcase.expectErr, errString(errs)); d != "" {
				t.Fatalf("CredentialProviderConfig validation mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func errString(errs errors.Aggregate) string {
	if errs != nil {
		return errs.Error()
	}
	return ""
}
