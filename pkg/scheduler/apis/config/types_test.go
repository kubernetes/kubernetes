/*
Copyright 2020 The Kubernetes Authors.

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

package config

import (
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestPluginsApply(t *testing.T) {
	tests := []struct {
		name            string
		customPlugins   *Plugins
		defaultPlugins  *Plugins
		expectedPlugins *Plugins
	}{
		{
			name: "AppendCustomPlugin",
			customPlugins: &Plugins{
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "CustomPlugin"},
					},
				},
			},
			defaultPlugins: &Plugins{
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &Plugins{
				QueueSort: &PluginSet{Enabled: []Plugin{}},
				PreFilter: &PluginSet{Enabled: []Plugin{}},
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
						{Name: "CustomPlugin"},
					},
				},
				PostFilter: &PluginSet{Enabled: []Plugin{}},
				Score:      &PluginSet{Enabled: []Plugin{}},
				Reserve:    &PluginSet{Enabled: []Plugin{}},
				Permit:     &PluginSet{Enabled: []Plugin{}},
				PreBind:    &PluginSet{Enabled: []Plugin{}},
				Bind:       &PluginSet{Enabled: []Plugin{}},
				PostBind:   &PluginSet{Enabled: []Plugin{}},
				Unreserve:  &PluginSet{Enabled: []Plugin{}},
			},
		},
		{
			name: "InsertAfterDefaultPlugins2",
			customPlugins: &Plugins{
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin2"},
					},
					Disabled: []Plugin{
						{Name: "DefaultPlugin2"},
					},
				},
			},
			defaultPlugins: &Plugins{
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &Plugins{
				QueueSort: &PluginSet{Enabled: []Plugin{}},
				PreFilter: &PluginSet{Enabled: []Plugin{}},
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin2"},
					},
				},
				PostFilter: &PluginSet{Enabled: []Plugin{}},
				Score:      &PluginSet{Enabled: []Plugin{}},
				Reserve:    &PluginSet{Enabled: []Plugin{}},
				Permit:     &PluginSet{Enabled: []Plugin{}},
				PreBind:    &PluginSet{Enabled: []Plugin{}},
				Bind:       &PluginSet{Enabled: []Plugin{}},
				PostBind:   &PluginSet{Enabled: []Plugin{}},
				Unreserve:  &PluginSet{Enabled: []Plugin{}},
			},
		},
		{
			name: "InsertBeforeAllPlugins",
			customPlugins: &Plugins{
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
					Disabled: []Plugin{
						{Name: "*"},
					},
				},
			},
			defaultPlugins: &Plugins{
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &Plugins{
				QueueSort: &PluginSet{Enabled: []Plugin{}},
				PreFilter: &PluginSet{Enabled: []Plugin{}},
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "CustomPlugin"},
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
				PostFilter: &PluginSet{Enabled: []Plugin{}},
				Score:      &PluginSet{Enabled: []Plugin{}},
				Reserve:    &PluginSet{Enabled: []Plugin{}},
				Permit:     &PluginSet{Enabled: []Plugin{}},
				PreBind:    &PluginSet{Enabled: []Plugin{}},
				Bind:       &PluginSet{Enabled: []Plugin{}},
				PostBind:   &PluginSet{Enabled: []Plugin{}},
				Unreserve:  &PluginSet{Enabled: []Plugin{}},
			},
		},
		{
			name: "ReorderDefaultPlugins",
			customPlugins: &Plugins{
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "DefaultPlugin2"},
						{Name: "DefaultPlugin1"},
					},
					Disabled: []Plugin{
						{Name: "*"},
					},
				},
			},
			defaultPlugins: &Plugins{
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "DefaultPlugin1"},
						{Name: "DefaultPlugin2"},
					},
				},
			},
			expectedPlugins: &Plugins{
				QueueSort: &PluginSet{Enabled: []Plugin{}},
				PreFilter: &PluginSet{Enabled: []Plugin{}},
				Filter: &PluginSet{
					Enabled: []Plugin{
						{Name: "DefaultPlugin2"},
						{Name: "DefaultPlugin1"},
					},
				},
				PostFilter: &PluginSet{Enabled: []Plugin{}},
				Score:      &PluginSet{Enabled: []Plugin{}},
				Reserve:    &PluginSet{Enabled: []Plugin{}},
				Permit:     &PluginSet{Enabled: []Plugin{}},
				PreBind:    &PluginSet{Enabled: []Plugin{}},
				Bind:       &PluginSet{Enabled: []Plugin{}},
				PostBind:   &PluginSet{Enabled: []Plugin{}},
				Unreserve:  &PluginSet{Enabled: []Plugin{}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			test.defaultPlugins.Apply(test.customPlugins)
			if d := cmp.Diff(test.expectedPlugins, test.defaultPlugins); d != "" {
				t.Fatalf("plugins mismatch (-want +got):\n%s", d)
			}
		})
	}
}
