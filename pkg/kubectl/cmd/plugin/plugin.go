/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package plugin

import (
	"fmt"
	"sync"

	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

type Plugin interface {
	// Name returns a unique plugin name for registration and lookup
	Name() string
	// ConfigOption (optional, may return nil) generates functional configuration options
	ConfigOption() cmdutil.ConfigOption
}

var (
	registered   = make(map[string]Plugin)
	activated    = make(map[string]struct{})
	guardPlugins sync.Mutex
)

func Register(p Plugin) error {
	name := p.Name()

	guardPlugins.Lock()
	defer guardPlugins.Unlock()

	if _, exists := registered[name]; exists {
		return fmt.Errorf("plugin already registered: %q", name)
	}
	registered[name] = p
	return nil
}

func Activate(pluginNames ...string) error {
	guardPlugins.Lock()
	defer guardPlugins.Unlock()

	for _, name := range pluginNames {
		if _, exists := registered[name]; exists {
			activated[name] = struct{}{}
		} else {
			return fmt.Errorf("cannot activate unregistered plugin %q", name)
		}
	}
	return nil
}

func ConfigOptions() (options []cmdutil.ConfigOption) {
	guardPlugins.Lock()
	defer guardPlugins.Unlock()

	for _, p := range registered {
		if _, exists := activated[p.Name()]; exists {
			opt := p.ConfigOption()
			if opt != nil {
				options = append(options, opt)
			}
		}
	}
	return
}
