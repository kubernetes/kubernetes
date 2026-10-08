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
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestModifyConfigWritesToFirstKubeconfigFile(t *testing.T) {
	const (
		contextNameA   = "context-a"
		contextNameB   = "context-b"
		newContextName = "new-context"
	)

	tempdir := t.TempDir()
	configFile1 := filepath.Join(tempdir, "kubeconfig-a")
	configFile2 := filepath.Join(tempdir, "kubeconfig-b")

	// The first kubeconfig has everything.
	err := os.WriteFile(configFile1, []byte(`
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
	err = os.WriteFile(configFile2, []byte(`
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
	envVarValue := fmt.Sprintf("%s%c%s", configFile2, filepath.ListSeparator, configFile1)
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
	config1, err := LoadFromFile(configFile1) // file sorts first, but was specified last
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if config1.CurrentContext != contextNameA {
		t.Errorf("Config should not be modified, but was. Expected %q, got %q", contextNameA, config1.CurrentContext)
	}

	config2, err := LoadFromFile(configFile2) // file sorts last, but was specified first
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if config2.CurrentContext != newContextName {
		t.Errorf("Config should be modified, but was not. Expected %q, got %q", newContextName, config2.CurrentContext)
	}
}

func TestModifyConfigWithReadOnlySource(t *testing.T) {
	for _, operation := range []string{"current-context", "new-context", "existing-context", "deleted-context", "read-only-context"} {
		t.Run(operation, func(t *testing.T) {
			dir := t.TempDir()
			readOnlyDir := filepath.Join(dir, "readonly")
			if err := os.Mkdir(readOnlyDir, 0755); err != nil {
				t.Fatal(err)
			}
			readOnlyFile := filepath.Join(readOnlyDir, "config")
			writableFile := filepath.Join(dir, "config")
			readOnlyConfig := clientcmdapi.NewConfig()
			readOnlyConfig.Contexts["shared"] = &clientcmdapi.Context{Namespace: "shared"}
			writableConfig := clientcmdapi.NewConfig()
			writableConfig.Contexts["local"] = &clientcmdapi.Context{Namespace: "old"}
			for filename, config := range map[string]*clientcmdapi.Config{readOnlyFile: readOnlyConfig, writableFile: writableConfig} {
				if err := WriteToFile(*config, filename); err != nil {
					t.Fatal(err)
				}
			}
			original, err := os.ReadFile(readOnlyFile)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(readOnlyDir, 0555); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := os.Chmod(readOnlyDir, 0755); err != nil {
					t.Error(err)
				}
			})
			// Windows and privileged users may not enforce directory mode bits.
			if err := lockFile(readOnlyFile); err == nil {
				if err := unlockFile(readOnlyFile); err != nil {
					t.Fatal(err)
				}
				t.Skip("directory permissions do not prevent lock creation")
			} else if !os.IsPermission(err) {
				t.Fatalf("expected permission error, got %v", err)
			}

			rules := &ClientConfigLoadingRules{Precedence: []string{writableFile, readOnlyFile}}
			config, err := rules.GetStartingConfig()
			if err != nil {
				t.Fatal(err)
			}
			switch operation {
			case "current-context":
				config.CurrentContext = "local"
			case "new-context":
				config.Contexts["new"] = clientcmdapi.NewContext()
				config.Contexts["new"].Namespace = "new"
			case "existing-context":
				config.Contexts["local"].Namespace = "new"
			case "deleted-context":
				delete(config.Contexts, "local")
			case "read-only-context":
				config.Contexts["shared"].Namespace = "new"
			}
			err = ModifyConfig(rules, *config, false)
			if operation == "read-only-context" {
				if !os.IsPermission(err) {
					t.Fatalf("expected permission error for unlocked destination, got %v", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				actual, err := rules.GetStartingConfig()
				if err != nil {
					t.Fatal(err)
				}
				if operation == "new-context" {
					config.Contexts["new"].LocationOfOrigin = writableFile
				}
				if !reflect.DeepEqual(actual, config) {
					t.Errorf("expected config %#v, got %#v", config, actual)
				}
			}
			unchanged, err := os.ReadFile(readOnlyFile)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(original, unchanged) {
				t.Error("read-only source was modified")
			}
			if _, err := os.Stat(lockName(writableFile)); !os.IsNotExist(err) {
				t.Errorf("destination lock was not removed: %v", err)
			}
		})
	}
}

func TestModifyConfigWithExistingLock(t *testing.T) {
	for _, lockedFile := range []string{"destination", "source"} {
		t.Run(lockedFile, func(t *testing.T) {
			dir := t.TempDir()
			destination := filepath.Join(dir, "a-config")
			source := filepath.Join(dir, "b-config")
			for _, filename := range []string{destination, source} {
				if err := WriteToFile(*clientcmdapi.NewConfig(), filename); err != nil {
					t.Fatal(err)
				}
			}
			filename := destination
			if lockedFile == "source" {
				filename = source
			}
			if err := lockFile(filename); err != nil {
				t.Fatal(err)
			}
			defer unlockFile(filename)
			rules := &ClientConfigLoadingRules{Precedence: []string{destination, source}}
			config := clientcmdapi.NewConfig()
			config.CurrentContext = "new"
			if err := ModifyConfig(rules, *config, false); !os.IsExist(err) {
				t.Fatalf("expected existing lock error, got %v", err)
			}
			actual, err := LoadFromFile(destination)
			if err != nil {
				t.Fatal(err)
			}
			if actual.CurrentContext != "" {
				t.Error("locked configuration was modified")
			}
			if _, err := os.Stat(lockName(filename)); err != nil {
				t.Errorf("existing lock was removed: %v", err)
			}
			if lockedFile == "source" {
				if _, err := os.Stat(lockName(destination)); !os.IsNotExist(err) {
					t.Errorf("destination lock was not removed after failure: %v", err)
				}
			}
		})
	}
}
