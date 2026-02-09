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
	"testing"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/kubectl/pkg/config/v1beta1"
	"sigs.k8s.io/yaml"
)

func TestViewOptions_Run(t *testing.T) {
	kubercContent := `apiVersion: kubectl.config.k8s.io/v1beta1
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

	expectedPref := &v1beta1.Preference{
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
		Aliases: []v1beta1.AliasOverride{
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

			// Unmarshal actual output to v1beta1.Preference
			var actualPref v1beta1.Preference
			if err := yaml.Unmarshal(out.Bytes(), &actualPref); err != nil {
				t.Fatalf("failed to unmarshal actual output: %v", err)
			}

			if diff := cmp.Diff(expectedPref, &actualPref); diff != "" {
				t.Errorf("Run() output mismatch (-expected +got):\n%s", diff)
			}
		})
	}
}
