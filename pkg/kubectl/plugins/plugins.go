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
	"strings"
)

// Plugin is the representation of a CLI extension (plugin).
type Plugin struct {
	Description
	Source
	Context RunningContext `json:"-"`
}

// PluginDescription holds everything needed to register a
// plugin as a command. Usually comes from a descriptor file.
type Description struct {
	Name      string    `json:"name"`
	ShortDesc string    `json:"shortDesc"`
	LongDesc  string    `json:"longDesc,omitempty"`
	Example   string    `json:"example,omitempty"`
	Command   string    `json:"command"`
	Tree      []*Plugin `json:"tree,omitempty"`
}

// PluginSource holds the location of a given plugin in the filesystem.
type Source struct {
	Dir            string `json:"-"`
	DescriptorName string `json:"-"`
}

var (
	IncompleteError  = fmt.Errorf("incomplete plugin descriptor: name, shortDesc and command fields are required")
	InvalidNameError = fmt.Errorf("plugin name can't contain spaces")
)

func (p Plugin) Validate() error {
	if len(p.Name) == 0 || len(p.ShortDesc) == 0 || (len(p.Command) == 0 && len(p.Tree) == 0) {
		return IncompleteError
	}
	if strings.Index(p.Name, " ") > -1 {
		return InvalidNameError
	}
	for _, child := range p.Tree {
		if err := child.Validate(); err != nil {
			return err
		}
	}
	return nil
}

func (p Plugin) IsValid() bool {
	return p.Validate() == nil
}

// Plugins is a list of plugins.
type Plugins []*Plugin
