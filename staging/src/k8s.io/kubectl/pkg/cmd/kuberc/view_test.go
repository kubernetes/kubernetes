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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	v1 "k8s.io/kubectl/pkg/config/v1"
	"sigs.k8s.io/yaml"
)

func TestViewOptions_Run(t *testing.T) {
	kubercContent := `apiVersion: kubectl.config.k8s.io/v1
kind: Preference
defaults:
- command: get
  options:
  - name: output
    default: wide
aliases:
- name: getn
  command: get
  prependArgs:
  - nodes
`

	expectedPref := &v1.Preference{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "kubectl.config.k8s.io/v1",
			Kind:       "Preference",
		},
		Defaults: []v1.CommandDefaults{
			{
				Command: "get",
				Options: []v1.CommandOptionDefault{
					{
						Name:    "output",
						Default: "wide",
					},
				},
			},
		},
		Aliases: []v1.AliasOverride{
			{
				Name:        "getn",
				Command:     "get",
				PrependArgs: []string{"nodes"},
			},
		},
	}

	tests := []struct {
		name         string
		outputFormat string
	}{
		{
			name:         "yaml output",
			outputFormat: "yaml",
		},
		{
			name:         "json output",
			outputFormat: "json",
		},
	}

	t.Run("declining to generate a default kuberc", func(t *testing.T) {
		tmpDir := t.TempDir()
		kubercPath := filepath.Join(tmpDir, "kuberc")

		streams, in, out, errOut := genericiooptions.NewTestIOStreams()
		in.WriteString("n\n")
		o := &ViewOptions{
			KubeRCFile: kubercPath,
			PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput("yaml"),
			IOStreams:  streams,
		}

		if err := o.Run(); err != nil {
			t.Fatalf("Run() unexpected error = %v", err)
		}
		if _, err := os.Stat(kubercPath); !os.IsNotExist(err) {
			t.Errorf("expected no kuberc file to be created, stat returned: %v", err)
		}
		if out.Len() != 0 {
			t.Errorf("expected empty stdout, got: %q", out.String())
		}
		if got := errOut.String(); !strings.Contains(got, "kuberc file not found") {
			t.Errorf("expected the missing-file notice on stderr, got: %q", got)
		}
	})

	t.Run("accepting to generate a default kuberc", func(t *testing.T) {
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				kubercPath := filepath.Join(t.TempDir(), "kuberc")
				streams, in, out, errOut := genericiooptions.NewTestIOStreams()
				in.WriteString("y\n")
				o := &ViewOptions{
					KubeRCFile: kubercPath,
					PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(tt.outputFormat),
					IOStreams:  streams,
				}

				if err := o.Run(); err != nil {
					t.Fatalf("Run() unexpected error = %v", err)
				}
				saved, err := os.ReadFile(kubercPath)
				if err != nil {
					t.Fatalf("read generated kuberc file: %v", err)
				}
				var savedPref v1.Preference
				if err := yaml.UnmarshalStrict(saved, &savedPref); err != nil {
					t.Fatalf("decode generated kuberc file: %v", err)
				}

				if tt.outputFormat == "json" && !json.Valid(out.Bytes()) {
					t.Fatalf("stdout is not valid JSON: %q", out.String())
				}
				var displayedPref v1.Preference
				if err := yaml.UnmarshalStrict(out.Bytes(), &displayedPref); err != nil {
					t.Fatalf("decode %s from stdout: %v\nstdout: %q", tt.outputFormat, err, out.String())
				}
				if diff := cmp.Diff(savedPref, displayedPref, cmpopts.EquateEmpty()); diff != "" {
					t.Errorf("preference mismatch (-saved +displayed):\n%s", diff)
				}

				wantMessages := fmt.Sprintf("kuberc file not found at %s\n"+
					"Would you like to generate a default kuberc file with recommended options? (y/N): Updated %s\n", kubercPath, kubercPath)
				if diff := cmp.Diff(wantMessages, errOut.String()); diff != "" {
					t.Errorf("stderr mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir, err := os.MkdirTemp("", "kuberc-test-")
			if err != nil {
				t.Fatalf("failed to create temp dir: %v", err)
			}
			defer func() {
				os.RemoveAll(tmpDir) // nolint:errcheck
			}()

			kubercPath := filepath.Join(tmpDir, "kuberc")
			if err := os.WriteFile(kubercPath, []byte(kubercContent), 0644); err != nil {
				t.Fatalf("failed to write kuberc file: %v", err)
			}

			streams, _, out, _ := genericiooptions.NewTestIOStreams()
			o := &ViewOptions{
				KubeRCFile: kubercPath,
				PrintFlags: genericclioptions.NewPrintFlags("").WithDefaultOutput(tt.outputFormat),
				IOStreams:  streams,
			}

			err = o.Run()
			if err != nil {
				t.Fatalf("Run() unexpected error = %v", err)
			}

			// Unmarshal actual output to v1.Preference
			var actualPref v1.Preference
			if err := yaml.Unmarshal(out.Bytes(), &actualPref); err != nil {
				t.Fatalf("failed to unmarshal actual output: %v", err)
			}

			if diff := cmp.Diff(expectedPref, &actualPref); diff != "" {
				t.Errorf("Run() output mismatch (-expected +got):\n%s", diff)
			}
		})
	}
}
