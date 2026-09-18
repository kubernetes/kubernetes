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

	"github.com/spf13/pflag"

	restclient "k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
)

func TestNewConfigFlagsDefaults(t *testing.T) {
	flags := NewConfigFlags(true)

	if flags.Insecure == nil || *flags.Insecure {
		t.Errorf("expected Insecure to default to false")
	}
	if flags.Timeout == nil || *flags.Timeout != "0" {
		t.Errorf("expected Timeout to default to %q, got %v", "0", flags.Timeout)
	}
	if flags.discoveryBurst != 300 {
		t.Errorf("expected discoveryBurst to default to 300, got %d", flags.discoveryBurst)
	}
	if !flags.usePersistentConfig {
		t.Errorf("expected usePersistentConfig to be true")
	}
	if flags.ImpersonateGroup == nil || len(*flags.ImpersonateGroup) != 0 {
		t.Errorf("expected ImpersonateGroup to default to an empty slice")
	}
}

func TestConfigFlagsAddFlagsSkipsNilFields(t *testing.T) {
	flags := &ConfigFlags{
		Namespace: ptr.To(""),
		Context:   ptr.To(""),
	}
	flagSet := pflag.NewFlagSet("test", pflag.ContinueOnError)
	flags.AddFlags(flagSet)

	if flagSet.Lookup(flagNamespace) == nil {
		t.Errorf("expected %q flag to be registered", flagNamespace)
	}
	if flagSet.Lookup(flagContext) == nil {
		t.Errorf("expected %q flag to be registered", flagContext)
	}
	if flagSet.Lookup(flagAPIServer) != nil {
		t.Errorf("expected %q flag to be skipped when field is nil", flagAPIServer)
	}
	if flagSet.Lookup(flagBearerToken) != nil {
		t.Errorf("expected %q flag to be skipped when field is nil", flagBearerToken)
	}
}

func TestConfigFlagsAddFlagsRegistersAllFields(t *testing.T) {
	flags := NewConfigFlags(false)
	flags.WithDeprecatedPasswordFlag()
	flagSet := pflag.NewFlagSet("test", pflag.ContinueOnError)
	flags.AddFlags(flagSet)

	shorthands := map[string]string{
		flagNamespace: "n",
		flagAPIServer: "s",
	}
	names := []string{
		flagClusterName, flagAuthInfoName, flagContext, flagNamespace, flagAPIServer,
		flagTLSServerName, flagInsecure, flagCertFile, flagKeyFile, flagCAFile,
		flagBearerToken, flagImpersonate, flagImpersonateUID, flagImpersonateGroup,
		flagImpersonateUserExtra, flagUsername, flagPassword, flagTimeout,
		flagDisableCompression, flagProxyURL, flagCacheDir, "kubeconfig",
	}
	for _, name := range names {
		f := flagSet.Lookup(name)
		if f == nil {
			t.Errorf("expected %q flag to be registered", name)
			continue
		}
		if want, ok := shorthands[name]; ok && f.Shorthand != want {
			t.Errorf("expected %q flag shorthand %q, got %q", name, want, f.Shorthand)
		}
	}
}

func TestConfigFlagsWithers(t *testing.T) {
	flags := NewConfigFlags(false)

	if got := flags.WithDiscoveryBurst(500); got.discoveryBurst != 500 {
		t.Errorf("expected discoveryBurst to be 500, got %d", got.discoveryBurst)
	}
	if got := flags.WithDiscoveryQPS(50.5); got.discoveryQPS != 50.5 {
		t.Errorf("expected discoveryQPS to be 50.5, got %v", got.discoveryQPS)
	}

	called := false
	flags.WithWrapConfigFn(func(c *restclient.Config) *restclient.Config {
		called = true
		return c
	})
	if flags.WrapConfigFn == nil {
		t.Fatal("expected WrapConfigFn to be set")
	}
	flags.WrapConfigFn(&restclient.Config{})
	if !called {
		t.Errorf("expected WrapConfigFn to be invoked")
	}

	flags.WithDeprecatedPasswordFlag()
	if flags.Username == nil || flags.Password == nil {
		t.Errorf("expected WithDeprecatedPasswordFlag to initialize Username and Password")
	}
}

func TestComputeDiscoverCacheDir(t *testing.T) {
	tests := []struct {
		name       string
		parentDir  string
		host       string
		wantSuffix string
	}{
		{
			name:       "https scheme is stripped",
			parentDir:  "/cache",
			host:       "https://example.com:6443",
			wantSuffix: "example.com_6443",
		},
		{
			name:       "http scheme is stripped",
			parentDir:  "/cache",
			host:       "http://example.com:6443",
			wantSuffix: "example.com_6443",
		},
		{
			name:       "illegal characters are collapsed",
			parentDir:  "/cache",
			host:       "https://[::1]:6443",
			wantSuffix: "___1__6443",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := computeDiscoverCacheDir(tc.parentDir, tc.host)
			want := filepath.Join(tc.parentDir, tc.wantSuffix)
			if got != want {
				t.Errorf("computeDiscoverCacheDir(%q, %q) = %q, want %q", tc.parentDir, tc.host, got, want)
			}
		})
	}
}

func TestGetDefaultCacheDir(t *testing.T) {
	t.Run("uses KUBECACHEDIR when set", func(t *testing.T) {
		t.Setenv("KUBECACHEDIR", "/custom/cache/dir")
		if got := getDefaultCacheDir(); got != "/custom/cache/dir" {
			t.Errorf("expected %q, got %q", "/custom/cache/dir", got)
		}
	})

	t.Run("falls back to home dir cache when unset", func(t *testing.T) {
		t.Setenv("KUBECACHEDIR", "")
		got := getDefaultCacheDir()
		want := filepath.Join(".kube", "cache")
		if filepath.Base(filepath.Dir(got)) != ".kube" || filepath.Base(got) != "cache" {
			t.Errorf("expected default cache dir to end with %q, got %q", want, got)
		}
	})
}

func TestToRawKubeConfigLoaderNamespaceOverride(t *testing.T) {
	emptyKubeConfig := filepath.Join(t.TempDir(), "config")
	const emptyConfigYAML = "apiVersion: v1\nkind: Config\nclusters: []\ncontexts: []\nusers: []\ncurrent-context: \"\"\n"
	if err := os.WriteFile(emptyKubeConfig, []byte(emptyConfigYAML), 0o600); err != nil {
		t.Fatalf("failed to write temp kubeconfig: %v", err)
	}

	flags := &ConfigFlags{
		Namespace:  ptr.To("my-namespace"),
		KubeConfig: ptr.To(emptyKubeConfig),
	}

	namespace, overridden, err := flags.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !overridden {
		t.Errorf("expected namespace to be marked as overridden")
	}
	if namespace != "my-namespace" {
		t.Errorf("expected namespace %q, got %q", "my-namespace", namespace)
	}
}
