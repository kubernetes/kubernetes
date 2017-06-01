/*
Copyright 2017 The Kubernetes Authors.

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

package plugins

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestSuccessfulDirectoryPluginLoader(t *testing.T) {
	tmp, err := setupValidPlugins(3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.RemoveAll(tmp)

	loader := &DirectoryPluginLoader{
		Directory: tmp,
	}
	plugins, err := loader.Load()
	if err != nil {
		t.Errorf("Unexpected error loading plugins: %v", err)
	}

	if count := len(plugins); count != 3 {
		t.Errorf("Unexpected number of loaded plugins, wanted 3, got %d", count)
	}

	for _, plugin := range plugins {
		if m, _ := regexp.MatchString("^plugin[123]$", plugin.Name); !m {
			t.Errorf("Unexpected plugin name %s", plugin.Name)
		}
		if m, _ := regexp.MatchString("^The plugin[123] test plugin$", plugin.ShortDesc); !m {
			t.Errorf("Unexpected plugin short desc %s", plugin.ShortDesc)
		}
		if m, _ := regexp.MatchString("^echo plugin[123]$", plugin.Command); !m {
			t.Errorf("Unexpected plugin command %s", plugin.Command)
		}
	}
}

func TestEmptyDirectoryPluginLoader(t *testing.T) {
	loader := &DirectoryPluginLoader{}
	_, err := loader.Load()
	if err == nil {
		t.Errorf("Expected error, got none")
	}
	if m, _ := regexp.MatchString("^directory not specified$", err.Error()); !m {
		t.Errorf("Unexpected error %v", err)
	}
}

func TestNotDirectoryPluginLoader(t *testing.T) {
	tmp, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("unexpected ioutil.TempDir error: %v", err)
	}
	defer os.RemoveAll(tmp)

	file := filepath.Join(tmp, "test.tmp")
	if err := ioutil.WriteFile(file, []byte("test"), 644); err != nil {
		t.Fatalf("unexpected ioutil.WriteFile error: %v", err)
	}

	loader := &DirectoryPluginLoader{
		Directory: file,
	}
	_, err = loader.Load()
	if err == nil {
		t.Errorf("Expected error, got none")
	}
	if !strings.Contains(err.Error(), "not a directory") {
		t.Errorf("Unexpected error %v", err)
	}
}

func TestUnexistentDirectoryPluginLoader(t *testing.T) {
	loader := &DirectoryPluginLoader{
		Directory: "/hopefully-does-not-exist",
	}
	_, err := loader.Load()
	if err == nil {
		t.Errorf("Expected error, got none")
	}
	if !strings.Contains(err.Error(), "no such file or directory") {
		t.Errorf("Unexpected error %v", err)
	}
}

func TestPluginsEnvVarPluginLoader(t *testing.T) {
	tmp, err := setupValidPlugins(1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.RemoveAll(tmp)

	env := "KUBECTL_PLUGINS_PATH"
	os.Setenv(env, tmp)
	defer os.Unsetenv(env)

	loader := PluginsEnvVarPluginLoader()

	plugins, err := loader.Load()
	if err != nil {
		t.Errorf("Unexpected error loading plugins: %v", err)
	}

	if count := len(plugins); count != 1 {
		t.Errorf("Unexpected number of loaded plugins, wanted 1, got %d", count)
	}

	plugin := plugins[0]
	if "plugin1" != plugin.Name {
		t.Errorf("Unexpected plugin name %s", plugin.Name)
	}
	if "The plugin1 test plugin" != plugin.ShortDesc {
		t.Errorf("Unexpected plugin short desc %s", plugin.ShortDesc)
	}
	if "echo plugin1" != plugin.Command {
		t.Errorf("Unexpected plugin command %s", plugin.Command)
	}
}

func TestIncompletePluginDescriptor(t *testing.T) {
	tmp, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("unexpected ioutil.TempDir error: %v", err)
	}

	descriptor := `
name: incomplete
shortDesc: The incomplete test plugin`

	if err := os.Mkdir(filepath.Join(tmp, "incomplete"), 0755); err != nil {
		t.Fatalf("unexpected os.Mkdir error: %v", err)
	}
	if err := ioutil.WriteFile(filepath.Join(tmp, "incomplete", "plugin.yaml"), []byte(descriptor), 0644); err != nil {
		t.Fatalf("unexpected ioutil.WriteFile error: %v", err)
	}

	defer os.RemoveAll(tmp)

	loader := &DirectoryPluginLoader{
		Directory: tmp,
	}
	plugins, err := loader.Load()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if count := len(plugins); count != 0 {
		t.Errorf("Unexpected number of loaded plugins, wanted 0, got %d", count)
	}
}

func setupValidPlugins(count int) (string, error) {
	tmp, err := ioutil.TempDir("", "")
	if err != nil {
		return "", fmt.Errorf("unexpected ioutil.TempDir error: %v", err)
	}

	for i := 1; i <= count; i++ {
		name := fmt.Sprintf("plugin%d", i)
		descriptor := fmt.Sprintf(`
name: %[1]s
shortDesc: The %[1]s test plugin
command: echo %[1]s`, name)

		if err := os.Mkdir(filepath.Join(tmp, name), 0755); err != nil {
			return "", fmt.Errorf("unexpected os.Mkdir error: %v", err)
		}
		if err := ioutil.WriteFile(filepath.Join(tmp, name, "plugin.yaml"), []byte(descriptor), 0644); err != nil {
			return "", fmt.Errorf("unexpected ioutil.WriteFile error: %v", err)
		}
	}

	return tmp, nil
}
