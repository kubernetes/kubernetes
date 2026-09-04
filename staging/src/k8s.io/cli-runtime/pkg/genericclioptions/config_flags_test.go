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

package genericclioptions

import (
	"os"
	"path/filepath"
	"testing"

	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
)

const testKubeConfig = `apiVersion: v1
kind: Config
clusters:
- name: test
  cluster:
    server: https://localhost:6443
    insecure-skip-tls-verify: true
contexts:
- name: test
  context:
    cluster: test
    user: test
current-context: test
users:
- name: test
  user:
    token: sometoken
`

func writeKubeConfig(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config")
	if err := os.WriteFile(path, []byte(testKubeConfig), 0600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestToRESTConfigResolvesOnceWithPersistentConfig(t *testing.T) {
	f := NewConfigFlags(true)
	f.KubeConfig = ptr.To(writeKubeConfig(t))

	first, err := f.ToRESTConfig()
	if err != nil {
		t.Fatal(err)
	}
	second, err := f.ToRESTConfig()
	if err != nil {
		t.Fatal(err)
	}

	if first == second {
		t.Fatal("expected each caller to get its own copy, got the same pointer")
	}
	if f.restConfig == nil {
		t.Fatal("expected the resolved config to be memoized")
	}
	if first == f.restConfig || second == f.restConfig {
		t.Fatal("expected callers to get copies, not the memoized config itself")
	}
	if first.Host != second.Host || first.BearerToken != second.BearerToken {
		t.Errorf("expected identical configs, got %q/%q", first.Host, second.Host)
	}
}

func TestToRESTConfigCopiesAreIndependent(t *testing.T) {
	f := NewConfigFlags(true)
	f.KubeConfig = ptr.To(writeKubeConfig(t))

	first, err := f.ToRESTConfig()
	if err != nil {
		t.Fatal(err)
	}
	// Callers routinely mutate these; the mutation must not leak into later callers.
	first.APIPath = "/apis"
	first.ContentConfig.GroupVersion = nil
	first.UserAgent = "mutated"

	second, err := f.ToRESTConfig()
	if err != nil {
		t.Fatal(err)
	}
	if second.APIPath == "/apis" || second.UserAgent == "mutated" {
		t.Errorf("mutation of one config leaked into the next: APIPath=%q UserAgent=%q", second.APIPath, second.UserAgent)
	}
}

func TestToRESTConfigDoesNotMemoizeWithoutPersistentConfig(t *testing.T) {
	f := NewConfigFlags(false)
	f.KubeConfig = ptr.To(writeKubeConfig(t))

	if _, err := f.ToRESTConfig(); err != nil {
		t.Fatal(err)
	}
	if f.restConfig != nil {
		t.Error("expected no memoization when usePersistentConfig is false")
	}
}

func TestToRESTConfigAppliesWrapConfigFnPerCall(t *testing.T) {
	f := NewConfigFlags(true)
	f.KubeConfig = ptr.To(writeKubeConfig(t))

	calls := 0
	f.WrapConfigFn = func(c *rest.Config) *rest.Config {
		calls++
		c.UserAgent = "wrapped"
		return c
	}

	for i := 0; i < 3; i++ {
		c, err := f.ToRESTConfig()
		if err != nil {
			t.Fatal(err)
		}
		if c.UserAgent != "wrapped" {
			t.Errorf("call %d: expected WrapConfigFn to be applied, got UserAgent=%q", i, c.UserAgent)
		}
	}
	if calls != 3 {
		t.Errorf("expected WrapConfigFn to run on every call, ran %d times", calls)
	}
	if f.restConfig.UserAgent == "wrapped" {
		t.Error("expected WrapConfigFn to mutate the copy, not the memoized config")
	}
}

func TestToRESTConfigPropagatesError(t *testing.T) {
	f := NewConfigFlags(true)
	f.KubeConfig = ptr.To(filepath.Join(t.TempDir(), "does-not-exist"))

	if _, err := f.ToRESTConfig(); err == nil {
		t.Fatal("expected an error for a missing kubeconfig")
	}
	if f.restConfig != nil {
		t.Error("expected nothing to be memoized after a failed resolve")
	}
	// A second call must still report the error rather than a memoized nil.
	if _, err := f.ToRESTConfig(); err == nil {
		t.Fatal("expected an error on the second call too")
	}
}
