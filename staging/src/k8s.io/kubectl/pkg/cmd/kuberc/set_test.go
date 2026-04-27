/*
Copyright The Kubernetes Authors.

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

package kuberc

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/kubectl/pkg/config/v1beta1"
	"sigs.k8s.io/yaml"
)

func TestSetOptions_Run_Defaults(t *testing.T) {
	tests := []struct {
		name           string
		existingKuberc string
		options        SetOptions
		expectedPref   *v1beta1.Preference
		expectError    bool
		errorContains  string
	}{
		{
			name:           "create new defaults",
			existingKuberc: "",
			options: SetOptions{
				Section: sectionDefaults,
				Command: "get",
				Options: []string{"output=wide"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Defaults: []v1beta1.CommandDefaults{
					{
						Command: "get",
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "wide",
							},
						},
					},
				},
			},
		},
		{
			name: "add defaults to existing file",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
defaults:
- command: get
  options:
  - name: output
    default: wide
`,
			options: SetOptions{
				Section: sectionDefaults,
				Command: "create",
				Options: []string{"output=yaml"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Defaults: []v1beta1.CommandDefaults{
					{
						Command: "get",
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "wide",
							},
						},
					},
					{
						Command: "create",
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "yaml",
							},
						},
					},
				},
			},
		},
		{
			name: "overwrite existing defaults",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
defaults:
- command: get
  options:
  - name: output
    default: wide
`,
			options: SetOptions{
				Section:   sectionDefaults,
				Command:   "get",
				Options:   []string{"output=json"},
				Overwrite: true,
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Defaults: []v1beta1.CommandDefaults{
					{
						Command: "get",
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "json",
							},
						},
					},
				},
			},
		},
		{
			name: "overwrite without options preserves existing options",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
defaults:
- command: get
  options:
  - name: output
    default: wide
`,
			options: SetOptions{
				Section:   sectionDefaults,
				Command:   "get",
				Options:   []string{}, // no options provided
				Overwrite: true,
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Defaults: []v1beta1.CommandDefaults{
					{
						Command: "get",
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "wide",
							},
						},
					},
				},
			},
		},
		{
			name: "overwrite with single option replaces all options",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
defaults:
- command: get
  options:
  - name: output
    default: wide
  - name: show-labels
    default: "true"
`,
			options: SetOptions{
				Section:   sectionDefaults,
				Command:   "get",
				Options:   []string{"output=json"},
				Overwrite: true,
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Defaults: []v1beta1.CommandDefaults{
					{
						Command: "get",
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "json",
							},
							// show-labels option is overwritten and thus not present
						},
					},
				},
			},
		},
		{
			name: "error without overwrite",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
defaults:
- command: get
  options:
  - name: output
    default: wide
`,
			options: SetOptions{
				Section:   sectionDefaults,
				Command:   "get",
				Options:   []string{"output=json"},
				Overwrite: false,
			},
			expectError:   true,
			errorContains: "already exist",
		},
		{
			name:           "subcommand with multiple options",
			existingKuberc: "",
			options: SetOptions{
				Section: sectionDefaults,
				Command: "set env",
				Options: []string{"output=yaml", "local=true"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Defaults: []v1beta1.CommandDefaults{
					{
						Command: "set env",
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "yaml",
							},
							{
								Name:    "local",
								Default: "true",
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir, err := os.MkdirTemp("", "kuberc-set-test-")
			if err != nil {
				t.Fatalf("failed to create temp dir: %v", err)
			}
			defer func() {
				os.RemoveAll(tmpDir) // nolint:errcheck
			}()

			kubercPath := filepath.Join(tmpDir, "kuberc")
			if tt.existingKuberc != "" {
				if err := os.WriteFile(kubercPath, []byte(tt.existingKuberc), 0644); err != nil {
					t.Fatalf("failed to write existing kuberc file: %v", err)
				}
			}

			streams, _, out, _ := genericiooptions.NewTestIOStreams()
			tt.options.KubeRCFile = kubercPath
			tt.options.IOStreams = streams

			err = tt.options.Run()

			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				if !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("expected error to contain %q, got: %v", tt.errorContains, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("Run() unexpected error = %v", err)
			}

			// Verify the file was written
			data, err := os.ReadFile(kubercPath)
			if err != nil {
				t.Fatalf("failed to read written kuberc file: %v", err)
			}

			var actualPref v1beta1.Preference
			if err := yaml.Unmarshal(data, &actualPref); err != nil {
				t.Fatalf("failed to unmarshal actual output: %v", err)
			}

			if diff := cmp.Diff(tt.expectedPref, &actualPref); diff != "" {
				t.Errorf("Run() output mismatch (-expected +got):\n%s", diff)
			}

			// Verify output message
			if !strings.Contains(out.String(), "Updated") {
				t.Errorf("expected output to contain 'Updated', got: %s", out.String())
			}
		})
	}
}

func TestSetOptions_Run_Aliases(t *testing.T) {
	tests := []struct {
		name           string
		existingKuberc string
		options        SetOptions
		expectedPref   *v1beta1.Preference
		expectError    bool
		errorContains  string
	}{
		{
			name:           "create new alias",
			existingKuberc: "",
			options: SetOptions{
				Section:     sectionAliases,
				AliasName:   "getn",
				Command:     "get",
				PrependArgs: []string{"nodes"},
				Options:     []string{"output=wide"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Aliases: []v1beta1.AliasOverride{
					{
						Name:        "getn",
						Command:     "get",
						PrependArgs: []string{"nodes"},
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "wide",
							},
						},
					},
				},
			},
		},
		{
			name: "add alias to existing file",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
aliases:
- name: getn
  command: get
  prependArgs:
  - nodes
`,
			options: SetOptions{
				Section:     sectionAliases,
				AliasName:   "getp",
				Command:     "get",
				PrependArgs: []string{"pods"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Aliases: []v1beta1.AliasOverride{
					{
						Name:        "getn",
						Command:     "get",
						PrependArgs: []string{"nodes"},
					},
					{
						Name:        "getp",
						Command:     "get",
						PrependArgs: []string{"pods"},
					},
				},
			},
		},
		{
			name: "overwrite existing alias",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
aliases:
- name: getn
  command: get
  prependArgs:
  - nodes
`,
			options: SetOptions{
				Section:     sectionAliases,
				AliasName:   "getn",
				Command:     "get",
				PrependArgs: []string{"namespaces"},
				Overwrite:   true,
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Aliases: []v1beta1.AliasOverride{
					{
						Name:        "getn",
						Command:     "get",
						PrependArgs: []string{"namespaces"},
					},
				},
			},
		},
		{
			name: "overwrite alias without options preserves existing options",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
aliases:
- name: getn
  command: get
  options:
  - name: output
    default: wide
`,
			options: SetOptions{
				Section:   sectionAliases,
				AliasName: "getn",
				Command:   "get",
				Options:   []string{}, // no options provided
				Overwrite: true,
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Aliases: []v1beta1.AliasOverride{
					{
						Name:        "getn",
						Command:     "get",
						PrependArgs: nil,
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "wide",
							},
						},
					},
				},
			},
		},
		{
			name: "overwrite alias without prependArgs preserves existing prependArgs",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
aliases:
- name: getn
  command: get
  prependArgs:
  - nodes
`,
			options: SetOptions{
				Section:     sectionAliases,
				AliasName:   "getn",
				Command:     "get",
				PrependArgs: []string{}, // no prependArgs provided
				Overwrite:   true,
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Aliases: []v1beta1.AliasOverride{
					{
						Name:        "getn",
						Command:     "get",
						PrependArgs: []string{"nodes"},
						Options:     nil,
					},
				},
			},
		},

		{
			name: "overwrite alias without appendArgs preserves existing appendArgs",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
aliases:
- name: getn
  command: get
  prependArgs:
  - nodes
  appendArgs:
  - --
  - custom-arg
`,
			options: SetOptions{
				Section:    sectionAliases,
				AliasName:  "getn",
				Command:    "get",
				AppendArgs: []string{}, // no appendArgs provided
				Overwrite:  true,
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Aliases: []v1beta1.AliasOverride{
					{
						Name:        "getn",
						Command:     "get",
						PrependArgs: []string{"nodes"},
						AppendArgs:  []string{"--", "custom-arg"},
						Options:     nil,
					},
				},
			},
		},
		{
			name: "update multiple fields simultaneously",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
aliases:
- name: getn
  command: get
  prependArgs:
  - nodes
  appendArgs:
  - --watch
  options:
  - name: output
    default: wide
`,
			options: SetOptions{
				Section:     sectionAliases,
				AliasName:   "getn",
				Command:     "get",
				PrependArgs: []string{"pods"},
				Options:     []string{"output=json"},
				AppendArgs:  []string{}, // no appendArgs provided
				Overwrite:   true,
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Aliases: []v1beta1.AliasOverride{
					{
						Name:        "getn",
						Command:     "get",
						PrependArgs: []string{"pods"},
						AppendArgs:  []string{"--watch"},
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "json",
							},
						},
					},
				},
			},
		},
		{
			name: "overwrite with single option replaces all options",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
aliases:
- name: getn
  command: get
  options:
  - name: output
    default: wide
  - name: show-labels
    default: "true"
`,
			options: SetOptions{
				Section:   sectionAliases,
				AliasName: "getn",
				Command:   "get",
				Options:   []string{"output=json"},
				Overwrite: true,
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Aliases: []v1beta1.AliasOverride{
					{
						Name:    "getn",
						Command: "get",
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "output",
								Default: "json",
							},
							// show-labels option is overwritten and thus not present
						},
					},
				},
			},
		},
		{
			name: "error without overwrite",
			existingKuberc: `apiVersion: kubectl.config.k8s.io/v1beta1
kind: Preference
aliases:
- name: getn
  command: get
  prependArgs:
  - nodes
`,
			options: SetOptions{
				Section:     sectionAliases,
				AliasName:   "getn",
				Command:     "get",
				PrependArgs: []string{"namespaces"},
				Overwrite:   false,
			},
			expectError:   true,
			errorContains: "already exists",
		},
		{
			name:           "alias with append args",
			existingKuberc: "",
			options: SetOptions{
				Section:    sectionAliases,
				AliasName:  "runx",
				Command:    "run",
				AppendArgs: []string{"--", "custom-arg1"},
				Options:    []string{"image=nginx"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				Aliases: []v1beta1.AliasOverride{
					{
						Name:       "runx",
						Command:    "run",
						AppendArgs: []string{"--", "custom-arg1"},
						Options: []v1beta1.CommandOptionDefault{
							{
								Name:    "image",
								Default: "nginx",
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir, err := os.MkdirTemp("", "kuberc-set-test-")
			if err != nil {
				t.Fatalf("failed to create temp dir: %v", err)
			}
			defer func() {
				os.RemoveAll(tmpDir) // nolint:errcheck
			}()

			kubercPath := filepath.Join(tmpDir, "kuberc")
			if tt.existingKuberc != "" {
				if err := os.WriteFile(kubercPath, []byte(tt.existingKuberc), 0644); err != nil {
					t.Fatalf("failed to write existing kuberc file: %v", err)
				}
			}

			streams, _, out, _ := genericiooptions.NewTestIOStreams()
			tt.options.KubeRCFile = kubercPath
			tt.options.IOStreams = streams

			err = tt.options.Run()

			if tt.expectError {
				if err == nil {
					t.Fatalf("expected error but got none")
				}
				if !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("expected error to contain %q, got: %v", tt.errorContains, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("Run() unexpected error = %v", err)
			}

			// Verify the file was written
			data, err := os.ReadFile(kubercPath)
			if err != nil {
				t.Fatalf("failed to read written kuberc file: %v", err)
			}

			var actualPref v1beta1.Preference
			if err := yaml.Unmarshal(data, &actualPref); err != nil {
				t.Fatalf("failed to unmarshal actual output: %v", err)
			}

			if diff := cmp.Diff(tt.expectedPref, &actualPref); diff != "" {
				t.Errorf("Run() output mismatch (-expected +got):\n%s", diff)
			}

			// Verify output message
			if !strings.Contains(out.String(), "Updated") {
				t.Errorf("expected output to contain 'Updated', got: %s", out.String())
			}
		})
	}
}

func TestSetOptions_Run_CredentialPlugin(t *testing.T) {
	tests := []struct {
		name           string
		existingKuberc string
		options        SetOptions
		expectedPref   *v1beta1.Preference
		expectError    bool
		errorContains  string
	}{
		{
			name:           "plugin policy AllowAll",
			existingKuberc: "",
			options: SetOptions{
				Section:      sectionCredentialPlugin,
				PluginPolicy: string(v1beta1.PluginPolicyAllowAll),
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyAllowAll,
			},
		},
		{
			name:           "plugin policy DenyAll",
			existingKuberc: "",
			options: SetOptions{
				Section:      sectionCredentialPlugin,
				PluginPolicy: string(v1beta1.PluginPolicyDenyAll),
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyDenyAll,
			},
		},
		{
			name:           "plugin policy Allowlist with single valid entry",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyAllowlist),
				AllowlistEntries: []string{"command=foobar"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyAllowlist,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: "foobar"},
				},
			},
		},
		{
			name:           "plugin policy Allowlist with multiple valid entries",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyAllowlist),
				AllowlistEntries: []string{"command=foobar", "command=barbaz"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyAllowlist,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: "foobar"},
					{Command: "barbaz"},
				},
			},
		},
		{
			name:           "plugin policy Allowlist with no entries",
			existingKuberc: "",
			options: SetOptions{
				Section:      sectionCredentialPlugin,
				PluginPolicy: string(v1beta1.PluginPolicyAllowlist),
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyAllowlist,
			},
			expectError: true,
		},
		{
			name:           "plugin policy Allowlist with one invalid entry",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyAllowlist),
				AllowlistEntries: []string{"calvinball=asdf"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyAllowlist,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: "hello"},
				},
			},
			expectError: true,
		},
		{
			name:           "plugin policy Allowlist with empty command",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyAllowlist),
				AllowlistEntries: []string{"command="},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyAllowlist,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: ""},
				},
			},
			expectError: true,
		},
		{
			name:           "plugin policy Allowlist with both valid and invalid entries",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyAllowlist),
				AllowlistEntries: []string{"command=hello", "calvinball=asdf"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyAllowlist,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: "hello"},
				},
			},
			expectError: true,
		},
		{
			name:           "invalid plugin policy",
			existingKuberc: "",
			options: SetOptions{
				Section:      sectionCredentialPlugin,
				PluginPolicy: "Foo",
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.CredentialPluginPolicy("Foo"),
			},
			expectError: true,
		},
		{
			name:           "empty plugin policy",
			existingKuberc: "",
			options: SetOptions{
				Section:      sectionCredentialPlugin,
				PluginPolicy: "",
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
			},
			expectError: true,
		},
		{
			name:           "allowlist entries with AllowAll policy",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyAllowAll),
				AllowlistEntries: []string{"command=foo"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyAllowAll,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: "foo"},
				},
			},
			expectError: true,
		},
		{
			name:           "allowlist entries with DenyAll policy",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyDenyAll),
				AllowlistEntries: []string{"command=foo"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyDenyAll,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: "foo"},
				},
			},
			expectError: true,
		},
		{
			name:           "use of deprecated name field",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyAllowlist),
				AllowlistEntries: []string{"name=foo"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyDenyAll,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: "foo"},
				},
			},
			expectError: true,
		},
		{
			name:           "improperly formatted allowlist entry",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyAllowlist),
				AllowlistEntries: []string{"command:foo"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyDenyAll,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: "foo"},
				},
			},
			expectError: true,
		},
		{
			name:           "improperly formatted allowlist entry",
			existingKuberc: "",
			options: SetOptions{
				Section:          sectionCredentialPlugin,
				PluginPolicy:     string(v1beta1.PluginPolicyAllowlist),
				AllowlistEntries: []string{"command=foo,command=bar"},
			},
			expectedPref: &v1beta1.Preference{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "kubectl.config.k8s.io/v1beta1",
					Kind:       "Preference",
				},
				CredentialPluginPolicy: v1beta1.PluginPolicyAllowlist,
				CredentialPluginAllowlist: []v1beta1.AllowlistEntry{
					{Command: "foo"},
					{Command: "bar"},
				},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir, err := os.MkdirTemp("", "kuberc-set-test-")
			if err != nil {
				t.Fatalf("failed to create temp dir: %v", err)
			}
			defer func() {
				os.RemoveAll(tmpDir) // nolint:errcheck
			}()

			kubercPath := filepath.Join(tmpDir, "kuberc")
			if tt.existingKuberc != "" {
				if err := os.WriteFile(kubercPath, []byte(tt.existingKuberc), 0644); err != nil {
					t.Fatalf("failed to write existing kuberc file: %v", err)
				}
			}

			streams, _, out, _ := genericiooptions.NewTestIOStreams()
			tt.options.KubeRCFile = kubercPath
			tt.options.IOStreams = streams

			validationErr := tt.options.Validate()
			runErr := tt.options.Run()
			if tt.expectError {
				if validationErr == nil && runErr == nil {
					t.Fatalf("expected error but got none")
				}

				firstErr := validationErr
				if firstErr == nil {
					firstErr = runErr
				}

				if !strings.Contains(firstErr.Error(), tt.errorContains) {
					t.Errorf("expected firstError to contain %q, got: %v", tt.errorContains, firstErr)
				}
				return
			}

			if err != nil {
				t.Fatalf("Run() unexpected error = %v", err)
			}

			// Verify the file was written
			data, err := os.ReadFile(kubercPath)
			if err != nil {
				t.Fatalf("failed to read written kuberc file: %v", err)
			}

			var actualPref v1beta1.Preference
			if err := yaml.Unmarshal(data, &actualPref); err != nil {
				t.Fatalf("failed to unmarshal actual output: %v", err)
			}

			if diff := cmp.Diff(tt.expectedPref, &actualPref); diff != "" {
				t.Errorf("Run() output mismatch (-expected +got):\n%s", diff)
			}

			// Verify output message
			if !strings.Contains(out.String(), "Updated") {
				t.Errorf("expected output to contain 'Updated', got: %s", out.String())
			}
		})
	}
}
