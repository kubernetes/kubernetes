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
	"reflect"
	"testing"
	"time"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestConfigFlagsBuildsDefaultClientConfigWithOverrides(t *testing.T) {
	flags := NewConfigFlags(false)
	*flags.Timeout = "30s"
	*flags.Impersonate = "alice"
	*flags.ImpersonateGroup = []string{"team-a", "team-b"}

	loader, ok := flags.ToRawKubeConfigLoader().(*clientConfig)
	if !ok {
		t.Fatalf("unexpected loader type %T", flags.ToRawKubeConfigLoader())
	}

	loadingRules, ok := loader.ConfigAccess().(*clientcmd.ClientConfigLoadingRules)
	if !ok {
		t.Fatalf("unexpected config access type %T", loader.ConfigAccess())
	}

	defaultConfig, err := loadingRules.DefaultClientConfig.ClientConfig()
	if err != nil {
		t.Fatalf("unexpected error building default config: %v", err)
	}

	if defaultConfig.Timeout != 30*time.Second {
		t.Fatalf("expected timeout %s, got %s", 30*time.Second, defaultConfig.Timeout)
	}
	if defaultConfig.Impersonate.UserName != "alice" {
		t.Fatalf("expected impersonation user %q, got %q", "alice", defaultConfig.Impersonate.UserName)
	}
	if !reflect.DeepEqual(defaultConfig.Impersonate.Groups, []string{"team-a", "team-b"}) {
		t.Fatalf("unexpected impersonation groups: %#v", defaultConfig.Impersonate.Groups)
	}

	expectedOverrides := &clientcmd.ConfigOverrides{
		ClusterDefaults: clientcmd.ClusterDefaults,
		AuthInfo: clientcmdapi.AuthInfo{
			Impersonate:       "alice",
			ImpersonateGroups: []string{"team-a", "team-b"},
		},
		Timeout: "30s",
	}
	expectedConfig, err := clientcmd.NewNonInteractiveClientConfig(*clientcmdapi.NewConfig(), "", expectedOverrides, nil).ClientConfig()
	if err != nil {
		t.Fatalf("unexpected error building override config: %v", err)
	}
	if !loadingRules.IsDefaultConfig(expectedConfig) {
		t.Fatal("expected override-aware config to remain comparable to the default loader config")
	}
}
