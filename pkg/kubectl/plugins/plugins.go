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

import "fmt"

// Plugin is the representation of a CLI extension (plugin).
type Plugin struct {
	PluginDescription
	PluginSource
	Context RunningContext `json:"-"`
}

// PluginDescription holds everything needed to register a
// plugin as a command. Usually comes from a descriptor file.
type PluginDescription struct {
	Name      string `json:"name"`
	ShortDesc string `json:"shortDesc"`
	LongDesc  string `json:"longDesc,omitempty"`
	Example   string `json:"example,omitempty"`
	Command   string `json:"command"`
}

// PluginSource holds the location of a given plugin in the filesystem.
type PluginSource struct {
	Dir            string `json:"-"`
	DescriptorName string `json:"-"`
}

func (p Plugin) Validate() error {
	if len(p.Name) == 0 || len(p.ShortDesc) == 0 || len(p.Command) == 0 {
		return fmt.Errorf("Incomplete plugin descriptor: Name, ShortDesc and Command fields are required")
	}
	return nil
}

// Plugins is a list of plugins.
type Plugins []*Plugin
