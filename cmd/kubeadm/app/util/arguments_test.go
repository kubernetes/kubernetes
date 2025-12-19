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

package util

import (
	"reflect"
	"sort"
	"testing"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func TestArgumentsToCommand(t *testing.T) {
	var tests = []struct {
		name      string
		base      []kubeadmapi.Arg
		overrides []kubeadmapi.Arg
		expected  []string
	}{
		{
			name: "override an argument from the base",
			base: []kubeadmapi.Arg{
				{Name: "admission-control", Value: "NamespaceLifecycle"},
				{Name: "allow-privileged", Value: "true"},
			},
			overrides: []kubeadmapi.Arg{
				{Name: "admission-control", Value: "NamespaceLifecycle,LimitRanger"},
			},
			expected: []string{
				"--allow-privileged=true",
				"--admission-control=NamespaceLifecycle,LimitRanger",
			},
		},
		{
			name: "override an argument from the base and add duplicate",
			base: []kubeadmapi.Arg{
				{Name: "token-auth-file", Value: "/token"},
				{Name: "tls-sni-cert-key", Value: "/some/path/"},
			},
			overrides: []kubeadmapi.Arg{
				{Name: "tls-sni-cert-key", Value: "/some/new/path"},
				{Name: "tls-sni-cert-key", Value: "/some/new/path/subpath"},
			},
			expected: []string{
				"--token-auth-file=/token",
				"--tls-sni-cert-key=/some/new/path",
				"--tls-sni-cert-key=/some/new/path/subpath",
			},
		},
		{
			name: "override all duplicate arguments from base",
			base: []kubeadmapi.Arg{
				{Name: "token-auth-file", Value: "/token"},
				{Name: "tls-sni-cert-key", Value: "foo"},
				{Name: "tls-sni-cert-key", Value: "bar"},
			},
			overrides: []kubeadmapi.Arg{
				{Name: "tls-sni-cert-key", Value: "/some/new/path"},
			},
			expected: []string{
				"--token-auth-file=/token",
				"--tls-sni-cert-key=/some/new/path",
			},
		},
		{
			name: "add an argument that is not in base",
			base: []kubeadmapi.Arg{
				{Name: "allow-privileged", Value: "true"},
			},
			overrides: []kubeadmapi.Arg{
				{Name: "admission-control", Value: "NamespaceLifecycle,LimitRanger"},
			},
			expected: []string{
				"--allow-privileged=true",
				"--admission-control=NamespaceLifecycle,LimitRanger",
			},
		},
		{
			name: "allow empty strings in base",
			base: []kubeadmapi.Arg{
				{Name: "allow-privileged", Value: "true"},
				{Name: "something-that-allows-empty-string", Value: ""},
			},
			overrides: []kubeadmapi.Arg{
				{Name: "admission-control", Value: "NamespaceLifecycle,LimitRanger"},
			},
			expected: []string{
				"--allow-privileged=true",
				"--something-that-allows-empty-string=",
				"--admission-control=NamespaceLifecycle,LimitRanger",
			},
		},
		{
			name: "allow empty strings in overrides",
			base: []kubeadmapi.Arg{
				{Name: "allow-privileged", Value: "true"},
				{Name: "something-that-allows-empty-string", Value: "foo"},
			},
			overrides: []kubeadmapi.Arg{
				{Name: "admission-control", Value: "NamespaceLifecycle,LimitRanger"},
				{Name: "something-that-allows-empty-string", Value: ""},
			},
			expected: []string{
				"--allow-privileged=true",
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--something-that-allows-empty-string=",
			},
		},
		{
			name: "base are sorted and overrides are not",
			base: []kubeadmapi.Arg{
				{Name: "b", Value: "true"},
				{Name: "c", Value: "true"},
				{Name: "a", Value: "true"},
			},
			overrides: []kubeadmapi.Arg{
				{Name: "e", Value: "true"},
				{Name: "b", Value: "true"},
				{Name: "d", Value: "true"},
			},
			expected: []string{
				"--a=true",
				"--c=true",
				"--e=true",
				"--b=true",
				"--d=true",
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := ArgumentsToCommand(rt.base, rt.overrides)
			if !reflect.DeepEqual(actual, rt.expected) {
				t.Errorf("failed ArgumentsToCommand:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
			}
		})
	}
}

func TestArgumentsFromCommand(t *testing.T) {
	var tests = []struct {
		name     string
		args     []string
		expected []kubeadmapi.Arg
	}{
		{
			name: "normal case",
			args: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
			},
			expected: []kubeadmapi.Arg{
				{Name: "admission-control", Value: "NamespaceLifecycle,LimitRanger"},
				{Name: "allow-privileged", Value: "true"},
			},
		},
		{
			name: "test that feature-gates is working",
			args: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
				"--feature-gates=EnableFoo=true,EnableBar=false",
			},
			expected: []kubeadmapi.Arg{
				{Name: "admission-control", Value: "NamespaceLifecycle,LimitRanger"},
				{Name: "allow-privileged", Value: "true"},
				{Name: "feature-gates", Value: "EnableFoo=true,EnableBar=false"},
			},
		},
		{
			name: "test that a binary can be the first arg",
			args: []string{
				"kube-apiserver",
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
				"--feature-gates=EnableFoo=true,EnableBar=false",
			},
			expected: []kubeadmapi.Arg{
				{Name: "admission-control", Value: "NamespaceLifecycle,LimitRanger"},
				{Name: "allow-privileged", Value: "true"},
				{Name: "feature-gates", Value: "EnableFoo=true,EnableBar=false"},
			},
		},
		{
			name: "allow duplicate args",
			args: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--tls-sni-cert-key=/some/path",
				"--tls-sni-cert-key=/some/path/subpath",
			},
			expected: []kubeadmapi.Arg{
				{Name: "admission-control", Value: "NamespaceLifecycle,LimitRanger"},
				{Name: "tls-sni-cert-key", Value: "/some/path"},
				{Name: "tls-sni-cert-key", Value: "/some/path/subpath"},
			},
		},
		{
			name: "args are sorted",
			args: []string{
				"--c=foo",
				"--a=foo",
				"--b=foo",
				"--b=bar",
			},
			expected: []kubeadmapi.Arg{
				{Name: "a", Value: "foo"},
				{Name: "b", Value: "bar"},
				{Name: "b", Value: "foo"},
				{Name: "c", Value: "foo"},
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			actual := ArgumentsFromCommand(rt.args)
			if !reflect.DeepEqual(actual, rt.expected) {
				t.Errorf("failed ArgumentsFromCommand:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
			}
		})
	}
}

func TestRoundtrip(t *testing.T) {
	var tests = []struct {
		name string
		args []string
	}{
		{
			name: "normal case",
			args: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
			},
		},
		{
			name: "test that feature-gates is working",
			args: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
				"--feature-gates=EnableFoo=true,EnableBar=false",
			},
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			// These two methods should be each other's opposite functions, test that by chaining the methods and see if you get the same result back
			actual := ArgumentsToCommand(ArgumentsFromCommand(rt.args), []kubeadmapi.Arg{})
			sort.Strings(actual)
			sort.Strings(rt.args)

			if !reflect.DeepEqual(actual, rt.args) {
				t.Errorf("failed TestRoundtrip:\nexpected:\n%v\nsaw:\n%v", rt.args, actual)
			}
		})
	}
}

func TestParseArgument(t *testing.T) {
	var tests = []struct {
		name        string
		arg         string
		expectedKey string
		expectedVal string
		expectedErr bool
	}{
		{
			name:        "arg cannot be empty",
			arg:         "",
			expectedErr: true,
		},
		{
			name:        "arg must contain -- and =",
			arg:         "a",
			expectedErr: true,
		},
		{
			name:        "arg must contain -- and =",
			arg:         "a-z",
			expectedErr: true,
		},
		{
			name:        "arg must contain --",
			arg:         "a=b",
			expectedErr: true,
		},
		{
			name:        "arg must contain a key",
			arg:         "--=b",
			expectedErr: true,
		},
		{
			name:        "arg can contain key but no value",
			arg:         "--a=",
			expectedKey: "a",
			expectedVal: "",
			expectedErr: false,
		},
		{
			name:        "simple case",
			arg:         "--a=b",
			expectedKey: "a",
			expectedVal: "b",
			expectedErr: false,
		},
		{
			name:        "keys/values with '-' should be supported",
			arg:         "--very-long-flag-name=some-value",
			expectedKey: "very-long-flag-name",
			expectedVal: "some-value",
			expectedErr: false,
		},
		{
			name:        "numbers should be handled correctly",
			arg:         "--some-number=0.2",
			expectedKey: "some-number",
			expectedVal: "0.2",
			expectedErr: false,
		},
		{
			name:        "lists should be handled correctly",
			arg:         "--admission-control=foo,bar,baz",
			expectedKey: "admission-control",
			expectedVal: "foo,bar,baz",
			expectedErr: false,
		},
		{
			name:        "more than one '=' should be allowed",
			arg:         "--feature-gates=EnableFoo=true,EnableBar=false",
			expectedKey: "feature-gates",
			expectedVal: "EnableFoo=true,EnableBar=false",
			expectedErr: false,
		},
	}

	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			key, val, actual := parseArgument(rt.arg)
			if (actual != nil) != rt.expectedErr {
				t.Errorf("failed parseArgument:\nexpected error:\n%t\nsaw error:\n%v", rt.expectedErr, actual)
			}
			if (key != rt.expectedKey) || (val != rt.expectedVal) {
				t.Errorf("failed parseArgument:\nexpected key: %s\nsaw key: %s\nexpected value: %s\nsaw value: %s", rt.expectedKey, key, rt.expectedVal, val)
			}
		})
	}
}
