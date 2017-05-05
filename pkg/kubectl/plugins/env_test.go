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
		env      Env
		expected string
	}{
		{
			env:      Env{"FOO", "BAR"},
			expected: "FOO=BAR",
		},
		{
			env:      Env{"FOO", "BAR="},
			expected: "FOO=BAR=",
		},
		{
			env:      Env{"FOO", ""},
			expected: "FOO=",
		},
	}
	for _, test := range tests {
		if s := test.env.String(); s != test.expected {
			t.Errorf("%v: expected string %q, got %q", test.env, test.expected, s)
		}
	}
}

func TestEnvListToSlice(t *testing.T) {
	tests := []struct {
		env      EnvList
		expected []string
	}{
		{
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
	for _, test := range tests {
		if s := test.env.Slice(); !reflect.DeepEqual(test.expected, s) {
			t.Errorf("%v: expected %v, got %v", test.env, test.expected, s)
		}
	}
}

func TestAddToEnvList(t *testing.T) {
	tests := []struct {
		add      []string
		expected EnvList
	}{
		{
			add: []string{"FOO=BAR", "EMPTY=", "EQUALS===", "JUSTNAME"},
			expected: EnvList{
				{"FOO", "BAR"},
				{"EMPTY", ""},
				{"EQUALS", "=="},
				{"JUSTNAME", ""},
			},
		},
	}
	for _, test := range tests {
		env := EnvList{}.Merge(test.add...)
		if !reflect.DeepEqual(test.expected, env) {
			t.Errorf("%v: expected %v, got %v", test.add, test.expected, env)
		}
	}
}

func TestFlagToEnv(t *testing.T) {
	flags := pflag.NewFlagSet("", pflag.ContinueOnError)
	flags.String("test", "ok", "")
	flags.String("kube-master", "http://something", "")
	flags.String("from-file", "default", "")
	flags.Parse([]string{"--from-file=nondefault"})

	tests := []struct {
		flag     *pflag.Flag
		prefix   string
		expected Env
	}{
		{
			flag:     flags.Lookup("test"),
			expected: Env{"TEST", "ok"},
		},
		{
			flag:     flags.Lookup("kube-master"),
			expected: Env{"KUBE_MASTER", "http://something"},
		},
		{
			prefix:   "KUBECTL_",
			flag:     flags.Lookup("from-file"),
			expected: Env{"KUBECTL_FROM_FILE", "nondefault"},
		},
	}
	for _, test := range tests {
		if env := FlagToEnv(test.flag, test.prefix); !reflect.DeepEqual(test.expected, env) {
			t.Errorf("%v: expected %v, got %v", test.flag.Name, test.expected, env)
		}
	}
}

func TestPluginDescriptorEnvProvider(t *testing.T) {
	tests := []struct {
		plugin   *Plugin
		expected EnvList
	}{
		{
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
	for _, test := range tests {
		provider := &PluginDescriptorEnvProvider{
			Plugin: test.plugin,
		}
		env, _ := provider.Env()
		if !reflect.DeepEqual(test.expected, env) {
			t.Errorf("%v: expected %v, got %v", test.plugin.Name, test.expected, env)
		}
	}

}
