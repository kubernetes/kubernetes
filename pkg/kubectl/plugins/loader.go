/*
Copyright 2016 The Kubernetes Authors.

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
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
)

type PluginLoader interface {
	Load() Plugins
}

type ConfigDirPluginLoader struct {
	pluginsDir string
}

func NewConfigDirPluginLoader() *ConfigDirPluginLoader {
	loader := &ConfigDirPluginLoader{}
	loader.pluginsDir = filepath.Join(clientcmd.RecommenderConfigDir, "plugins")
	return loader
}

func (l *ConfigDirPluginLoader) Load() (Plugins, error) {
	if len(l.pluginsDir) == 0 {
		return nil, fmt.Errorf("Directory not specified for the plugin loader.")
	}

	list := Plugins{}

	err := filepath.Walk(l.pluginsDir, func(path string, fileInfo os.FileInfo, err error) error {
		glog.V(9).Infof("Checking plugin in %s...", path)

		if l.pluginsDir == path ||
			fileInfo.IsDir() ||
			!strings.HasPrefix(fileInfo.Name(), "kubectl-") ||
			fileInfo.Mode()&0111 == 0 {
			return nil
		}

		cmd := exec.Command(path, "metadata")
		var out bytes.Buffer
		cmd.Stdout = &out

		glog.V(6).Infof("File %s is potentially a plugin, checking metadata with args %s...", cmd.Path, cmd.Args)

		if err := cmd.Run(); err == nil {
			plugin := &Plugin{
				Path: path,
			}

			if err := json.Unmarshal(out.Bytes(), plugin); err == nil {
				if len(plugin.Use) == 0 || len(plugin.Short) == 0 {
					glog.V(5).Infoln("Plugin metadata requires the 'use' and 'short' fields")
				}

				list = append(list, plugin)
				glog.V(5).Infof("Plugin loaded: %s", plugin)
			}
		}

		return nil
	})

	return list, err
}
