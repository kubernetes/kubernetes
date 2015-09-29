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

package testapi

import (
	"testing"
)

func TestResourcePathWithPrefix(t *testing.T) {
	testCases := []struct {
		prefix    string
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"prefix", "resource", "mynamespace", "myresource", "/api/" + Default.Version() + "/prefix/namespaces/mynamespace/resource/myresource"},
		{"prefix", "resource", "", "myresource", "/api/" + Default.Version() + "/prefix/resource/myresource"},
		{"prefix", "resource", "mynamespace", "", "/api/" + Default.Version() + "/prefix/namespaces/mynamespace/resource"},
		{"prefix", "resource", "", "", "/api/" + Default.Version() + "/prefix/resource"},
		{"", "resource", "mynamespace", "myresource", "/api/" + Default.Version() + "/namespaces/mynamespace/resource/myresource"},
	}
	for _, item := range testCases {
		if actual := Default.ResourcePathWithPrefix(item.prefix, item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for prefix: %s, resource: %s, namespace: %s and name: %s", item.expected, actual, item.prefix, item.resource, item.namespace, item.name)
		}
	}
}

func TestResourcePath(t *testing.T) {
	testCases := []struct {
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"resource", "mynamespace", "myresource", "/api/" + Default.Version() + "/namespaces/mynamespace/resource/myresource"},
		{"resource", "", "myresource", "/api/" + Default.Version() + "/resource/myresource"},
		{"resource", "mynamespace", "", "/api/" + Default.Version() + "/namespaces/mynamespace/resource"},
		{"resource", "", "", "/api/" + Default.Version() + "/resource"},
	}
	for _, item := range testCases {
		if actual := Default.ResourcePath(item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
		}
	}
}
