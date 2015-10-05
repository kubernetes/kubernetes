/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package remote

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

var (
	// ErrNotFound plugin not found
	ErrNotFound = errors.New("Plugin not found")
	PluginsPath = "/usr/lib/kubernetes/plugins"
)

// Registry defines behavior of a registry of plugins.
type Registry interface {
	// Plugins lists all plugins.
	Plugins() ([]*Plugin, error)
	// Plugin returns the plugin registered with the given name (or returns an error).
	Plugin(name string) (*Plugin, error)
}

// LocalRegistry defines a registry that is local (using unix socket).
type LocalRegistry struct{}

func newLocalRegistry() LocalRegistry {
	return LocalRegistry{}
}

// Plugin returns the plugin registered with the given name (or returns an error).
func (l *LocalRegistry) Plugin(name string) (*Plugin, error) {
	socketpaths := pluginPaths(PluginsPath, name, ".sock")

	for _, p := range socketpaths {
		if fi, err := os.Stat(p); err == nil && fi.Mode()&os.ModeSocket != 0 {
			return newLocalPlugin(name, "unix://"+p), nil
		}
	}

	var txtspecpaths []string
	txtspecpaths = append(txtspecpaths, pluginPaths(PluginsPath, name, ".spec")...)
	txtspecpaths = append(txtspecpaths, pluginPaths(PluginsPath, name, ".json")...)

	for _, p := range txtspecpaths {
		if _, err := os.Stat(p); err == nil {
			if strings.HasSuffix(p, ".json") {
				return readPluginJSONInfo(name, p)
			}
			return readPluginInfo(name, p)
		}
	}
	return nil, ErrNotFound
}

func readPluginInfo(name, path string) (*Plugin, error) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	addr := strings.TrimSpace(string(content))

	u, err := url.Parse(addr)
	if err != nil {
		return nil, err
	}

	if len(u.Scheme) == 0 {
		return nil, fmt.Errorf("Unknown protocol")
	}

	return newLocalPlugin(name, addr), nil
}

func readPluginJSONInfo(name, path string) (*Plugin, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var p Plugin
	if err := json.NewDecoder(f).Decode(&p); err != nil {
		return nil, err
	}
	p.Name = name
	if len(p.TLSConfig.CAFile) == 0 {
		p.TLSConfig.InsecureSkipVerify = true
	}

	return &p, nil
}

func pluginPaths(base, name, ext string) []string {
	return []string{
		filepath.Join(base, name+ext),
		filepath.Join(base, name, name+ext),
	}
}
