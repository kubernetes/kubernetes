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

func TestResourcePathWithPrefixForV1Beta3(t *testing.T) {
	if Version() != "v1beta3" {
		// Skip the test if we are not testing v1beta3.
		return
	}

	testCases := []struct {
		prefix    string
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"prefix", "resource", "mynamespace", "myresource", "/api/v1beta3/prefix/namespaces/mynamespace/resource/myresource"},
		{"prefix", "resource", "", "myresource", "/api/v1beta3/prefix/resource/myresource"},
		{"prefix", "resource", "mynamespace", "", "/api/v1beta3/prefix/namespaces/mynamespace/resource"},
		{"prefix", "resource", "", "", "/api/v1beta3/prefix/resource"},
		{"", "resource", "mynamespace", "myresource", "/api/v1beta3/namespaces/mynamespace/resource/myresource"},
	}
	for _, item := range testCases {
		if actual := ResourcePathWithPrefix(item.prefix, item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for prefix: %s, resource: %s, namespace: %s and name: %s", item.expected, actual, item.prefix, item.resource, item.namespace, item.name)
		}
	}
}

func TestResourcePathWithPrefixForV1Beta1(t *testing.T) {
	if Version() != "v1beta1" {
		// Skip the test if we are not testing v1beta1.
		return
	}

	testCases := []struct {
		prefix    string
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"prefix", "resource", "mynamespace", "myresource", "/api/v1beta1/prefix/resource/myresource"},
		{"prefix", "resource", "", "myresource", "/api/v1beta1/prefix/resource/myresource"},
		{"prefix", "resource", "mynamespace", "", "/api/v1beta1/prefix/resource"},
		{"prefix", "resource", "", "", "/api/v1beta1/prefix/resource"},
		{"", "resource", "mynamespace", "myresource", "/api/v1beta1/resource/myresource"},
	}
	for _, item := range testCases {
		if actual := ResourcePathWithPrefix(item.prefix, item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for prefix: %s, resource: %s, namespace: %s and name: %s", item.expected, actual, item.prefix, item.resource, item.namespace, item.name)
		}
	}
}

func TestResourcePathForV1Beta3(t *testing.T) {
	if Version() != "v1beta3" {
		// Skip the test if we are not testing v1beta3.
		return
	}

	testCases := []struct {
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"resource", "mynamespace", "myresource", "/api/v1beta3/namespaces/mynamespace/resource/myresource"},
		{"resource", "", "myresource", "/api/v1beta3/resource/myresource"},
		{"resource", "mynamespace", "", "/api/v1beta3/namespaces/mynamespace/resource"},
		{"resource", "", "", "/api/v1beta3/resource"},
	}
	for _, item := range testCases {
		if actual := ResourcePath(item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
		}
	}
}

func TestResourcePathForV1Beta1(t *testing.T) {
	if Version() != "v1beta1" {
		// Skip the test if we are not testing v1beta1.
		return
	}

	testCases := []struct {
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"resource", "mynamespace", "myresource", "/api/v1beta1/resource/myresource"},
		{"resource", "", "myresource", "/api/v1beta1/resource/myresource"},
		{"resource", "mynamespace", "", "/api/v1beta1/resource"},
		{"resource", "", "", "/api/v1beta1/resource"},
	}
	for _, item := range testCases {
		if actual := ResourcePath(item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
		}
	}
}

func TestResourcePathWithNamespaceQueryForV1Beta3(t *testing.T) {
	if Version() != "v1beta3" {
		// Skip the test if we are not testing v1beta3.
		return
	}

	testCases := []struct {
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"resource", "mynamespace", "myresource", "/api/v1beta3/namespaces/mynamespace/resource/myresource"},
		{"resource", "", "myresource", "/api/v1beta3/resource/myresource"},
		{"resource", "mynamespace", "", "/api/v1beta3/namespaces/mynamespace/resource"},
		{"resource", "", "", "/api/v1beta3/resource"},
	}
	for _, item := range testCases {
		if actual := ResourcePathWithNamespaceQuery(item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
		}
	}
}

func TestResourcePathWithNamespaceQueryForV1Beta1(t *testing.T) {
	if Version() != "v1beta1" {
		// Skip the test if we are not testing v1beta1.
		return
	}

	testCases := []struct {
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"resource", "mynamespace", "myresource", "/api/v1beta1/resource/myresource?namespace=mynamespace"},
		{"resource", "", "myresource", "/api/v1beta1/resource/myresource"},
		{"resource", "mynamespace", "", "/api/v1beta1/resource?namespace=mynamespace"},
		{"resource", "", "", "/api/v1beta1/resource"},
	}
	for _, item := range testCases {
		if actual := ResourcePathWithNamespaceQuery(item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
		}
	}
}
