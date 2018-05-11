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

package testing

import (
	"encoding/json"
	"fmt"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestResourcePathWithPrefix(t *testing.T) {
	testGroup := Test
	testGroup.externalGroupVersion.Group = "TestGroup"
	testCases := []struct {
		test      TestGroup
		prefix    string
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{Test, "prefix", "resource", "mynamespace", "myresource", "/api/" + Test.externalGroupVersion.Version + "/prefix/namespaces/mynamespace/resource/myresource"},
		{Test, "prefix", "resource", "", "myresource", "/api/" + Test.externalGroupVersion.Version + "/prefix/resource/myresource"},
		{Test, "prefix", "resource", "mynamespace", "", "/api/" + Test.externalGroupVersion.Version + "/prefix/namespaces/mynamespace/resource"},
		{Test, "prefix", "resource", "", "", "/api/" + Test.externalGroupVersion.Version + "/prefix/resource"},
		{Test, "", "resource", "mynamespace", "myresource", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
		{testGroup, "prefix", "resource", "mynamespace", "myresource", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/prefix/namespaces/mynamespace/resource/myresource"},
		{testGroup, "prefix", "resource", "", "myresource", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/prefix/resource/myresource"},
		{testGroup, "prefix", "resource", "mynamespace", "", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/prefix/namespaces/mynamespace/resource"},
		{testGroup, "prefix", "resource", "", "", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/prefix/resource"},
		{testGroup, "", "resource", "mynamespace", "myresource", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
	}
	for i, item := range testCases {
		name := fmt.Sprintf("%v/%v", i, item.test.externalGroupVersion.Group)
		t.Run(name, func(t *testing.T) {
			if actual := item.test.ResourcePathWithPrefix(item.prefix, item.resource, item.namespace, item.name); actual != item.expected {
				t.Errorf("Expected: %s, got: %s for prefix: %s, resource: %s, namespace: %s and name: %s", item.expected, actual, item.prefix, item.resource, item.namespace, item.name)
			}
		})
	}
}

func TestResourcePath(t *testing.T) {
	testGroup := Test
	testGroup.externalGroupVersion.Group = "TestGroup"
	testCases := []struct {
		test      TestGroup
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{Test, "resource", "mynamespace", "myresource", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
		{Test, "resource", "", "myresource", "/api/" + Test.externalGroupVersion.Version + "/resource/myresource"},
		{Test, "resource", "mynamespace", "", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource"},
		{Test, "resource", "", "", "/api/" + Test.externalGroupVersion.Version + "/resource"},
		{testGroup, "resource", "mynamespace", "myresource", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
		{testGroup, "resource", "", "myresource", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/resource/myresource"},
		{testGroup, "resource", "mynamespace", "", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource"},
		{testGroup, "resource", "", "", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/resource"},
	}
	for i, item := range testCases {
		name := fmt.Sprintf("%v/%v", i, item.test.externalGroupVersion.Group)
		t.Run(name, func(t *testing.T) {
			if actual := item.test.ResourcePath(item.resource, item.namespace, item.name); actual != item.expected {
				t.Errorf("Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
			}
		})
	}
}

func TestSubResourcePath(t *testing.T) {
	testGroup := Test
	testGroup.externalGroupVersion.Group = "TestGroup"
	testCases := []struct {
		test      TestGroup
		resource  string
		namespace string
		name      string
		sub       string
		expected  string
	}{
		{Test, "resource", "mynamespace", "myresource", "subresource", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource/subresource"},
		{Test, "resource", "mynamespace", "myresource", "", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
		{testGroup, "resource", "mynamespace", "myresource", "subresource", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource/subresource"},
		{testGroup, "resource", "mynamespace", "myresource", "", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
	}
	for i, item := range testCases {
		name := fmt.Sprintf("%v/%v", i, item.test.externalGroupVersion.Group)
		t.Run(name, func(t *testing.T) {
			if actual := item.test.SubResourcePath(item.resource, item.namespace, item.name, item.sub); actual != item.expected {
				t.Errorf("Expected: %s, got: %s for resource: %s, namespace: %s, name: %s and sub: %s", item.expected, actual, item.resource, item.namespace, item.name, item.sub)
			}
		})
	}
}

func TestSelfLink(t *testing.T) {
	testGroup := Test
	testGroup.externalGroupVersion.Group = "TestGroup"
	testCases := []struct {
		test     TestGroup
		resource string
		name     string
		expected string
	}{
		{Test, "resource", "name", "/api/" + Test.externalGroupVersion.Version + "/resource/name"},
		{Test, "resource", "", "/api/" + Test.externalGroupVersion.Version + "/resource"},
		{testGroup, "resource", "name", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/resource/name"},
		{testGroup, "resource", "", "/apis/" + testGroup.externalGroupVersion.Group + "/" + testGroup.externalGroupVersion.Version + "/resource"},
	}
	for i, item := range testCases {
		name := fmt.Sprintf("%v/%v", i, item.test.externalGroupVersion.Group)
		t.Run(name, func(t *testing.T) {
			if actual := item.test.SelfLink(item.resource, item.name); actual != item.expected {
				t.Errorf("Expected: %s, got: %s for resource: %s and name: %s", item.expected, actual, item.resource, item.name)
			}
		})
	}
}

var status = &metav1.Status{
	Status:  metav1.StatusFailure,
	Code:    200,
	Reason:  metav1.StatusReasonUnknown,
	Message: "",
}

func TestV1EncodeDecodeStatus(t *testing.T) {
	v1Codec := Test.Codec()

	encoded, err := runtime.Encode(v1Codec, status)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	typeMeta := metav1.TypeMeta{}
	if err := json.Unmarshal(encoded, &typeMeta); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if typeMeta.Kind != "Status" {
		t.Errorf("Kind is not set to \"Status\". Got %v", string(encoded))
	}
	if typeMeta.APIVersion != "v1" {
		t.Errorf("APIVersion is not set to \"v1\". Got %v", string(encoded))
	}
	decoded, err := runtime.Decode(v1Codec, encoded)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(status, decoded) {
		t.Errorf("expected: %#v, got: %#v", status, decoded)
	}
}
