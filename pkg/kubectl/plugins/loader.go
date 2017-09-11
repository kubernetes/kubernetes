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

	"github.com/ghodss/yaml"
	"github.com/golang/glog"

	"k8s.io/client-go/tools/clientcmd"
)

const PluginDescriptorFilename = "plugin.yaml"

// PluginLoader is capable of loading a list of plugin descriptions.
type PluginLoader interface {
	Load() (Plugins, error)
}

// DirectoryPluginLoader is a PluginLoader that loads plugin descriptions
// from a given directory in the filesystem. Plugins are located in subdirs
// under the loader "root", where each subdir must contain, at least, a plugin
// descriptor file called "plugin.yaml" that translates into a PluginDescription.
type DirectoryPluginLoader struct {
	Directory string
}

// Load reads the directory the loader holds and loads plugin descriptions.
func (l *DirectoryPluginLoader) Load() (Plugins, error) {
	if len(l.Directory) == 0 {
		return nil, fmt.Errorf("directory not specified")
	}

	list := Plugins{}

	stat, err := os.Stat(l.Directory)
	if err != nil {
		return nil, err
	}
	if !stat.IsDir() {
		return nil, fmt.Errorf("not a directory: %s", l.Directory)
	}

	base, err := filepath.Abs(l.Directory)
	if err != nil {
		return nil, err
	}

	// read the base directory tree searching for plugin descriptors
	// fails silently (descriptors unable to be read or unmarshalled are logged but skipped)
	err = filepath.Walk(base, func(path string, fileInfo os.FileInfo, walkErr error) error {
		if walkErr != nil || fileInfo.IsDir() || fileInfo.Name() != PluginDescriptorFilename {
			return nil
		}

		file, err := ioutil.ReadFile(path)
		if err != nil {
			glog.V(1).Infof("Unable to read plugin descriptor %s: %v", path, err)
			return nil
		}

		plugin := &Plugin{}
		if err := yaml.Unmarshal(file, plugin); err != nil {
			glog.V(1).Infof("Unable to unmarshal plugin descriptor %s: %v", path, err)
			return nil
		}

		if err := plugin.Validate(); err != nil {
			glog.V(1).Infof("%v", err)
			return nil
		}

		var setSource func(path string, fileInfo os.FileInfo, p *Plugin)
		setSource = func(path string, fileInfo os.FileInfo, p *Plugin) {
			p.Dir = filepath.Dir(path)
			p.DescriptorName = fileInfo.Name()
			for _, child := range p.Tree {
				setSource(path, fileInfo, child)
			}
		}
		setSource(path, fileInfo, plugin)

		glog.V(6).Infof("Plugin loaded: %s", plugin.Name)
		list = append(list, plugin)

		return nil
	})

	return list, err
}

// UserDirPluginLoader is a PluginLoader that loads plugins from the
// "plugins" directory under the user's kubeconfig dir (usually "~/.kube/plugins/").
func UserDirPluginLoader() PluginLoader {
	dir := filepath.Join(clientcmd.RecommendedConfigDir, "plugins")
	return &DirectoryPluginLoader{
		Directory: dir,
	}
}

// PathFromEnvVarPluginLoader is a PluginLoader that loads plugins from one or more
// directories specified by the provided env var name. In case the env var is not
// set, the PluginLoader just loads nothing. A list of subdirectories can be provided,
// which will be appended to each path specified by the env var.
func PathFromEnvVarPluginLoader(envVarName string, subdirs ...string) PluginLoader {
	env := os.Getenv(envVarName)
	if len(env) == 0 {
		return &DummyPluginLoader{}
	}
	loader := MultiPluginLoader{}
	for _, path := range filepath.SplitList(env) {
		dir := append([]string{path}, subdirs...)
		loader = append(loader, &DirectoryPluginLoader{
			Directory: filepath.Join(dir...),
		})
	}
	return loader
}

// PluginsEnvVarPluginLoader is a PluginLoader that loads plugins from one or more
// directories specified by the KUBECTL_PLUGINS_PATH env var.
func PluginsEnvVarPluginLoader() PluginLoader {
	return PathFromEnvVarPluginLoader("KUBECTL_PLUGINS_PATH")
}

// XDGDataPluginLoader is a PluginLoader that loads plugins from one or more
// directories specified by the XDG system directory structure spec in the
// XDG_DATA_DIRS env var, plus the "kubectl/plugins/" suffix. According to the
// spec, if XDG_DATA_DIRS is not set it defaults to "/usr/local/share:/usr/share".
func XDGDataPluginLoader() PluginLoader {
	envVarName := "XDG_DATA_DIRS"
	if len(os.Getenv(envVarName)) > 0 {
		return PathFromEnvVarPluginLoader(envVarName, "kubectl", "plugins")
	}
	return TolerantMultiPluginLoader{
		&DirectoryPluginLoader{
			Directory: "/usr/local/share/kubectl/plugins",
		},
		&DirectoryPluginLoader{
			Directory: "/usr/share/kubectl/plugins",
		},
	}
}

// MultiPluginLoader is a PluginLoader that can encapsulate multiple plugin loaders,
// a successful loading means every encapsulated loader was able to load without errors.
type MultiPluginLoader []PluginLoader

func (l MultiPluginLoader) Load() (Plugins, error) {
	plugins := Plugins{}
	for _, loader := range l {
		loaded, err := loader.Load()
		if err != nil {
			return nil, err
		}
		plugins = append(plugins, loaded...)
	}
	return plugins, nil
}

// TolerantMultiPluginLoader is a PluginLoader than encapsulates multiple plugins loaders,
// but is tolerant to errors while loading from them.
type TolerantMultiPluginLoader []PluginLoader

func (l TolerantMultiPluginLoader) Load() (Plugins, error) {
	plugins := Plugins{}
	for _, loader := range l {
		loaded, _ := loader.Load()
		if loaded != nil {
			plugins = append(plugins, loaded...)
		}
	}
	return plugins, nil
}

// DummyPluginLoader loads nothing.
type DummyPluginLoader struct{}

func (l *DummyPluginLoader) Load() (Plugins, error) {
	return Plugins{}, nil
}
