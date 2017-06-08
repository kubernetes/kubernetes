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

package cmd

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/util/version"
)

func TestDefaultAuthorizationModes(t *testing.T) {
	var tests = []struct {
		authzModes []string
		version    string
		expected   []string
	}{
		{[]string{"RBAC"}, "v1.6.0", []string{"RBAC"}},
		{[]string{"RBAC", "ABAC"}, "v1.6.4", []string{"RBAC", "ABAC"}},
		{[]string{"RBAC", "ABAC"}, "v1.7.0-beta.0", []string{"RBAC", "ABAC"}},
		{[]string{"RBAC"}, "v1.7.0", []string{"Node", "RBAC"}},
		{[]string{"RBAC", "Webhook"}, "v1.7.0-beta.1", []string{"Node", "RBAC", "Webhook"}},
		{[]string{"RBAC", "Webhook", "Node"}, "v1.7.0", []string{"RBAC", "Webhook", "Node"}},
		{[]string{"Node", "RBAC", "Webhook"}, "v1.7.0", []string{"Node", "RBAC", "Webhook"}},
	}
	for _, rt := range tests {
		k8sVersion, err := version.ParseSemantic(rt.version)
		if err != nil {
			t.Fatalf("couldn't parse given version")
		}
		actual := defaultAuthorizationModes(rt.authzModes, k8sVersion)
		if strings.Join(actual, ",") != strings.Join(rt.expected, ",") {
			t.Errorf(
				"failed TestDefaultAuthorizationModes:\n\texpected: %s\n\t  actual: %s",
				strings.Join(rt.expected, ","),
				strings.Join(actual, ","),
			)
		}
	}
}
