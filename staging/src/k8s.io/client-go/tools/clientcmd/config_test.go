/*
Copyright 2025 The Kubernetes Authors.

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

package clientcmd

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestModifyConfigWritesToFirstKubeconfigFile(t *testing.T) {
	const (
		contextNameA   = "context-a"
		contextNameB   = "context-b"
		newContextName = "new-context"
	)

	tempdir := t.TempDir()
	configFile1, _ := os.Create(filepath.Join(tempdir, "kubeconfig-a"))
	configFile2, _ := os.Create(filepath.Join(tempdir, "kubeconfig-b"))

	// The first kubeconfig has everything.
	err := os.WriteFile(configFile1.Name(), []byte(`
kind: Config
apiVersion: v1
clusters:
- cluster:
    api-version: v1
    server: https://kubernetes.default.svc:443
    certificate-authority: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
  name: kubeconfig-cluster
contexts:
- context:
    cluster: kubeconfig-cluster
    namespace: default
    user: kubeconfig-user
  name: `+contextNameA+`
current-context: `+contextNameA+`
users:
- name: kubeconfig-user
  user:
    tokenFile: /var/run/secrets/kubernetes.io/serviceaccount/token
`), os.FileMode(0755))

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// The second kubeconfig declares a new context and activates it.
	err = os.WriteFile(configFile2.Name(), []byte(`
kind: Config
apiVersion: v1
contexts:
- context:
    cluster: kubeconfig-cluster
    namespace: a-different-namespace
    user: kubeconfig-user
  name: `+contextNameB+`
current-context: `+contextNameB+`
`), os.FileMode(0755))

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Set KUBECONFIG to the files, in descending alphabetical order.
	// This will be used to check that they don't get sorted.
	envVarValue := fmt.Sprintf("%s%c%s", configFile2.Name(), filepath.ListSeparator, configFile1.Name())
	t.Setenv(RecommendedConfigPathEnvVar, envVarValue)

	// Load the kubeconfigs, change the active context, and call ModifyConfig.
	loadingRules := NewDefaultClientConfigLoadingRules()
	config, err := loadingRules.Load()

	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	newConfig := config.DeepCopy()
	newConfig.CurrentContext = newContextName
	err = ModifyConfig(loadingRules, *newConfig, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Load the files again and check that only configFile2 was changed.
	config1, err := LoadFromFile(configFile1.Name()) // file sorts first, but was specified last
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if config1.CurrentContext != contextNameA {
		t.Errorf("Config should not be modified, but was. Expected %q, got %q", contextNameA, config1.CurrentContext)
	}

	config2, err := LoadFromFile(configFile2.Name()) // file sorts last, but was specified first
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if config2.CurrentContext != newContextName {
		t.Errorf("Config should be modified, but was not. Expected %q, got %q", newContextName, config2.CurrentContext)
	}
}
