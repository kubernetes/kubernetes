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
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestPluginsNames(t *testing.T) {
	tests := []struct {
		name    string
		plugins *Plugins
		want    []string
	}{
		{
			name: "empty",
		},
		{
			name: "with duplicates",
			plugins: &Plugins{
				Filter: PluginSet{
					Enabled: []Plugin{
						{Name: "CustomFilter"},
					},
				},
				PreFilter: PluginSet{
					Enabled: []Plugin{
						{Name: "CustomFilter"},
					},
				},
				Score: PluginSet{
					Enabled: []Plugin{
						{Name: "CustomScore"},
					},
				},
			},
			want: []string{"CustomFilter", "CustomScore"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if d := cmp.Diff(test.want, test.plugins.Names()); d != "" {
				t.Fatalf("plugins mismatch (-want +got):\n%s", d)
			}
		})
	}
}

func TestPluginsNamesAllExtensionPoints(t *testing.T) {
	plugins := &Plugins{}
	val := reflect.ValueOf(plugins).Elem()

	want := sets.New[string]()
	for i := 0; i < val.NumField(); i++ {
		name := val.Type().Field(i).Name
		val.Field(i).Set(reflect.ValueOf(PluginSet{
			Enabled: []Plugin{{Name: name}},
		}))
		want.Insert(name)
	}

	gotNames := plugins.Names()
	if diff := cmp.Diff(sets.List(want), gotNames); diff != "" {
		t.Errorf("plugins.Names() doesn't contain all extension points (-want,+got):\n%s", diff)
	}
}
