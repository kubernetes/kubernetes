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
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
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
				"--tls-sni-cert-key=/some/new/path",
				"--tls-sni-cert-key=/some/new/path/subpath",
				"--token-auth-file=/token",
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
				"--tls-sni-cert-key=/some/new/path",
				"--token-auth-file=/token",
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
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
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
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
				"--something-that-allows-empty-string=",
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
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--allow-privileged=true",
				"--something-that-allows-empty-string=",
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
