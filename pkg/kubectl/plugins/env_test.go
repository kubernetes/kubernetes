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
	"reflect"
	"testing"

	"github.com/spf13/pflag"
)

func TestEnv(t *testing.T) {
	tests := []struct {
		name     string
		env      Env
		expected string
	}{
		{
			name:     "test1",
			env:      Env{"FOO", "BAR"},
			expected: "FOO=BAR",
		},
		{
			name:     "test2",
			env:      Env{"FOO", "BAR="},
			expected: "FOO=BAR=",
		},
		{
			name:     "test3",
			env:      Env{"FOO", ""},
			expected: "FOO=",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if s := tt.env.String(); s != tt.expected {
				t.Errorf("%v: expected string %q, got %q", tt.env, tt.expected, s)
			}
		})
	}
}

func TestEnvListToSlice(t *testing.T) {
	tests := []struct {
		name     string
		env      EnvList
		expected []string
	}{
		{
			name: "test1",
			env: EnvList{
				{"FOO", "BAR"},
				{"ZEE", "YO"},
				{"ONE", "1"},
				{"EQUALS", "=="},
				{"EMPTY", ""},
			},
			expected: []string{"FOO=BAR", "ZEE=YO", "ONE=1", "EQUALS===", "EMPTY="},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if s := tt.env.Slice(); !reflect.DeepEqual(tt.expected, s) {
				t.Errorf("%v: expected %v, got %v", tt.env, tt.expected, s)
			}
		})
	}
}

func TestAddToEnvList(t *testing.T) {
	tests := []struct {
		name     string
		add      []string
		expected EnvList
	}{
		{
			name: "test1",
			add:  []string{"FOO=BAR", "EMPTY=", "EQUALS===", "JUSTNAME"},
			expected: EnvList{
				{"FOO", "BAR"},
				{"EMPTY", ""},
				{"EQUALS", "=="},
				{"JUSTNAME", ""},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := EnvList{}.Merge(tt.add...)
			if !reflect.DeepEqual(tt.expected, env) {
				t.Errorf("%v: expected %v, got %v", tt.add, tt.expected, env)
			}
		})
	}
}

func TestFlagToEnv(t *testing.T) {
	flags := pflag.NewFlagSet("", pflag.ContinueOnError)
	flags.String("test", "ok", "")
	flags.String("kube-master", "http://something", "")
	flags.String("from-file", "default", "")
	flags.Parse([]string{"--from-file=nondefault"})

	tests := []struct {
		name     string
		flag     *pflag.Flag
		prefix   string
		expected Env
	}{
		{
			name:     "test1",
			flag:     flags.Lookup("test"),
			expected: Env{"TEST", "ok"},
		},
		{
			name:     "test2",
			flag:     flags.Lookup("kube-master"),
			expected: Env{"KUBE_MASTER", "http://something"},
		},
		{
			name:     "test3",
			prefix:   "KUBECTL_",
			flag:     flags.Lookup("from-file"),
			expected: Env{"KUBECTL_FROM_FILE", "nondefault"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if env := FlagToEnv(tt.flag, tt.prefix); !reflect.DeepEqual(tt.expected, env) {
				t.Errorf("%v: expected %v, got %v", tt.flag.Name, tt.expected, env)
			}
		})
	}
}

func TestPluginDescriptorEnvProvider(t *testing.T) {
	tests := []struct {
		name     string
		plugin   *Plugin
		expected EnvList
	}{
		{
			name: "test1",
			plugin: &Plugin{
				Description: Description{
					Name:      "test",
					ShortDesc: "Short Description",
					Command:   "foo --bar",
				},
			},
			expected: EnvList{
				{"KUBECTL_PLUGINS_DESCRIPTOR_NAME", "test"},
				{"KUBECTL_PLUGINS_DESCRIPTOR_SHORT_DESC", "Short Description"},
				{"KUBECTL_PLUGINS_DESCRIPTOR_LONG_DESC", ""},
				{"KUBECTL_PLUGINS_DESCRIPTOR_EXAMPLE", ""},
				{"KUBECTL_PLUGINS_DESCRIPTOR_COMMAND", "foo --bar"},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := &PluginDescriptorEnvProvider{
				Plugin: tt.plugin,
			}
			env, _ := provider.Env()
			if !reflect.DeepEqual(tt.expected, env) {
				t.Errorf("%v: expected %v, got %v", tt.plugin.Name, tt.expected, env)
			}
		})
	}

}
