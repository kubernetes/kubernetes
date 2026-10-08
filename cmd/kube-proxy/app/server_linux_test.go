//go:build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/pflag"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
)

func Test_platformApplyDefaults(t *testing.T) {
	testCases := []struct {
		name                string
		mode                kubeproxyconfig.ProxyMode
		expectedMode        kubeproxyconfig.ProxyMode
		detectLocal         kubeproxyconfig.LocalMode
		expectedDetectLocal kubeproxyconfig.LocalMode
	}{
		{
			name:                "defaults",
			mode:                "",
			expectedMode:        kubeproxyconfig.ProxyModeIPTables,
			detectLocal:         "",
			expectedDetectLocal: kubeproxyconfig.LocalModeClusterCIDR,
		},
		{
			name:                "explicit",
			mode:                kubeproxyconfig.ProxyModeIPTables,
			expectedMode:        kubeproxyconfig.ProxyModeIPTables,
			detectLocal:         kubeproxyconfig.LocalModeClusterCIDR,
			expectedDetectLocal: kubeproxyconfig.LocalModeClusterCIDR,
		},
		{
			name:                "override mode",
			mode:                "ipvs",
			expectedMode:        kubeproxyconfig.ProxyModeIPVS,
			detectLocal:         "",
			expectedDetectLocal: kubeproxyconfig.LocalModeClusterCIDR,
		},
		{
			name:                "override detect-local",
			mode:                "",
			expectedMode:        kubeproxyconfig.ProxyModeIPTables,
			detectLocal:         "NodeCIDR",
			expectedDetectLocal: kubeproxyconfig.LocalModeNodeCIDR,
		},
		{
			name:                "override both",
			mode:                "ipvs",
			expectedMode:        kubeproxyconfig.ProxyModeIPVS,
			detectLocal:         "NodeCIDR",
			expectedDetectLocal: kubeproxyconfig.LocalModeNodeCIDR,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := NewOptions()
			config := &kubeproxyconfig.KubeProxyConfiguration{
				Mode:            tc.mode,
				DetectLocalMode: tc.detectLocal,
			}

			options.platformApplyDefaults(config)
			if config.Mode != tc.expectedMode {
				t.Fatalf("expected mode: %s, but got: %s", tc.expectedMode, config.Mode)
			}
			if config.DetectLocalMode != tc.expectedDetectLocal {
				t.Fatalf("expected detect-local: %s, but got: %s", tc.expectedDetectLocal, config.DetectLocalMode)
			}
		})
	}
}

// Configuration is a startup snapshot: file updates must not change the running
// proxy's configuration. A replacement process loads the updated contents.
func TestConfigChangeRequiresRestart(t *testing.T) {
	const initial = `apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
bindAddress: 0.0.0.0
logging:
  verbosity: 0
`
	updated := strings.Replace(initial, "verbosity: 0", "verbosity: 1", 1)
	for _, mutation := range []string{"direct write", "atomic replacement", "ConfigMap projection"} {
		t.Run(mutation, func(t *testing.T) {
			dir := t.TempDir()
			configFile := filepath.Join(dir, "config.conf")
			write := func(path, content string) {
				t.Helper()
				if err := os.WriteFile(path, []byte(content), 0644); err != nil {
					t.Fatal(err)
				}
			}
			if mutation == "ConfigMap projection" {
				oldDir := filepath.Join(dir, "..old")
				if err := os.Mkdir(oldDir, 0755); err != nil {
					t.Fatal(err)
				}
				write(filepath.Join(oldDir, "config.conf"), initial)
				if err := os.Symlink("..old", filepath.Join(dir, "..data")); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink("..data/config.conf", configFile); err != nil {
					t.Fatal(err)
				}
			} else {
				write(configFile, initial)
			}
			load := func() *Options {
				t.Helper()
				opt := NewOptions()
				opt.ConfigFile = configFile
				if err := opt.Complete(new(pflag.FlagSet)); err != nil {
					t.Fatal(err)
				}
				if err := opt.Validate(); err != nil {
					t.Fatal(err)
				}
				return opt
			}
			opt := load()
			if opt.config.Logging.Verbosity != 0 {
				t.Fatalf("initial verbosity = %d, want 0", opt.config.Logging.Verbosity)
			}
			switch mutation {
			case "direct write":
				write(configFile, updated)
			case "atomic replacement":
				replacement := filepath.Join(dir, "config.tmp")
				write(replacement, updated)
				if err := os.Rename(replacement, configFile); err != nil {
					t.Fatal(err)
				}
			case "ConfigMap projection":
				newDir := filepath.Join(dir, "..new")
				if err := os.Mkdir(newDir, 0755); err != nil {
					t.Fatal(err)
				}
				write(filepath.Join(newDir, "config.conf"), updated)
				if err := os.Symlink("..new", filepath.Join(dir, "..data_tmp")); err != nil {
					t.Fatal(err)
				}
				if err := os.Rename(filepath.Join(dir, "..data_tmp"), filepath.Join(dir, "..data")); err != nil {
					t.Fatal(err)
				}
				if err := os.RemoveAll(filepath.Join(dir, "..old")); err != nil {
					t.Fatal(err)
				}
			}
			if opt.config.Logging.Verbosity != 0 {
				t.Fatalf("running config changed without restart: verbosity = %d", opt.config.Logging.Verbosity)
			}
			if restarted := load(); restarted.config.Logging.Verbosity != 1 {
				t.Fatalf("verbosity after restart = %d, want 1", restarted.config.Logging.Verbosity)
			}
		})
	}
}
