/*
Copyright 2026 The Kubernetes Authors.

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

package cmd

import (
	"strings"
	"testing"

	"k8s.io/client-go/tools/clientcmd/api"

	"k8s.io/cli-runtime/pkg/genericiooptions"
)

func TestGenerateContextName(t *testing.T) {
	tests := map[string]struct {
		context  *api.Context
		expected string
	}{
		"namespace only": {
			context:  &api.Context{Namespace: "foo"},
			expected: "foo",
		},
		"namespace and cluster": {
			context:  &api.Context{Namespace: "foo", Cluster: "my-cluster"},
			expected: "foo/my-cluster",
		},
		"namespace, cluster and auth info": {
			context:  &api.Context{Namespace: "foo", Cluster: "my-cluster", AuthInfo: "my-user"},
			expected: "foo/my-cluster/my-user",
		},
		"auth info with a slash gets truncated to its first segment": {
			context:  &api.Context{Namespace: "foo", Cluster: "my-cluster", AuthInfo: "my-user/my-cluster"},
			expected: "foo/my-cluster/my-user",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			actual := generateContextName(test.context)
			if actual != test.expected {
				t.Errorf("expected %q, got %q", test.expected, actual)
			}
		})
	}
}

func TestIsContextEqual(t *testing.T) {
	base := &api.Context{Cluster: "c", Namespace: "n", AuthInfo: "a"}

	tests := map[string]struct {
		a, b     *api.Context
		expected bool
	}{
		"equal contexts": {
			a:        base,
			b:        &api.Context{Cluster: "c", Namespace: "n", AuthInfo: "a"},
			expected: true,
		},
		"different namespace": {
			a:        base,
			b:        &api.Context{Cluster: "c", Namespace: "other", AuthInfo: "a"},
			expected: false,
		},
		"different cluster": {
			a:        base,
			b:        &api.Context{Cluster: "other", Namespace: "n", AuthInfo: "a"},
			expected: false,
		},
		"different auth info": {
			a:        base,
			b:        &api.Context{Cluster: "c", Namespace: "n", AuthInfo: "other"},
			expected: false,
		},
		"nil a": {
			a:        nil,
			b:        base,
			expected: false,
		},
		"nil b": {
			a:        base,
			b:        nil,
			expected: false,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			actual := isContextEqual(test.a, test.b)
			if actual != test.expected {
				t.Errorf("expected %v, got %v", test.expected, actual)
			}
		})
	}
}

func TestNamespaceOptionsValidate(t *testing.T) {
	tests := map[string]struct {
		currentContext string
		args           []string
		expectErr      string
	}{
		"no current context": {
			currentContext: "",
			args:           nil,
			expectErr:      errNoContext.Error(),
		},
		"too many args": {
			currentContext: "ctx",
			args:           []string{"a", "b"},
			expectErr:      "either one or no arguments are allowed",
		},
		"valid with one arg": {
			currentContext: "ctx",
			args:           []string{"a"},
			expectErr:      "",
		},
		"valid with no args": {
			currentContext: "ctx",
			args:           nil,
			expectErr:      "",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			o := &NamespaceOptions{
				rawConfig: api.Config{CurrentContext: test.currentContext},
				args:      test.args,
			}

			err := o.Validate()
			if test.expectErr == "" {
				if err != nil {
					t.Fatalf("expected no error, got %v", err)
				}
				return
			}

			if err == nil || err.Error() != test.expectErr {
				t.Fatalf("expected error %q, got %v", test.expectErr, err)
			}
		})
	}
}

func TestNamespaceOptionsRunCurrentNamespace(t *testing.T) {
	streams, _, out, _ := genericiooptions.NewTestIOStreams()

	o := &NamespaceOptions{
		IOStreams: streams,
		rawConfig: api.Config{
			CurrentContext: "ctx",
			Contexts: map[string]*api.Context{
				"ctx": {Namespace: "my-namespace"},
			},
		},
	}

	if err := o.Run(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := strings.TrimSpace(out.String()); got != "my-namespace" {
		t.Errorf("expected output %q, got %q", "my-namespace", got)
	}
}

func TestNamespaceOptionsRunCurrentNamespaceEmpty(t *testing.T) {
	streams, _, _, _ := genericiooptions.NewTestIOStreams()

	o := &NamespaceOptions{
		IOStreams: streams,
		rawConfig: api.Config{
			CurrentContext: "ctx",
			Contexts: map[string]*api.Context{
				"ctx": {Namespace: ""},
			},
		},
	}

	err := o.Run()
	if err == nil {
		t.Fatal("expected an error, got none")
	}
}

func TestNamespaceOptionsRunListNamespaces(t *testing.T) {
	streams, _, out, _ := genericiooptions.NewTestIOStreams()

	o := &NamespaceOptions{
		IOStreams:      streams,
		listNamespaces: true,
		rawConfig: api.Config{
			CurrentContext: "ctx",
			Contexts: map[string]*api.Context{
				"ctx":   {Namespace: "my-namespace"},
				"other": {Namespace: "other-namespace"},
				"empty": {Namespace: ""},
			},
		},
	}

	if err := o.Run(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := out.String()
	for _, expected := range []string{"my-namespace", "other-namespace"} {
		if !strings.Contains(got, expected) {
			t.Errorf("expected output to contain %q, got %q", expected, got)
		}
	}
}
