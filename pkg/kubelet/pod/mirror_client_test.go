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

package pod

import (
	"testing"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

func TestParsePodFullName(t *testing.T) {
	type nameTuple struct {
		Name      string
		Namespace string
	}
	successfulCases := map[string]nameTuple{
		"bar_foo":         {Name: "bar", Namespace: "foo"},
		"bar.org_foo.com": {Name: "bar.org", Namespace: "foo.com"},
		"bar-bar_foo":     {Name: "bar-bar", Namespace: "foo"},
	}
	failedCases := []string{"barfoo", "bar_foo_foo", ""}

	for podFullName, expected := range successfulCases {
		name, namespace, err := kubecontainer.ParsePodFullName(podFullName)
		if err != nil {
			t.Errorf("unexpected error when parsing the full name: %v", err)
			continue
		}
		if name != expected.Name || namespace != expected.Namespace {
			t.Errorf("expected name %q, namespace %q; got name %q, namespace %q",
				expected.Name, expected.Namespace, name, namespace)
		}
	}
	for _, podFullName := range failedCases {
		_, _, err := kubecontainer.ParsePodFullName(podFullName)
		if err == nil {
			t.Errorf("expected error when parsing the full name, got none")
		}
	}
}
