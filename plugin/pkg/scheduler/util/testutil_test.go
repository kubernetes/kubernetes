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
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestResourcePathWithPrefix(t *testing.T) {
	testCases := []struct {
		prefix    string
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"prefix", "resource", "mynamespace", "myresource", "/api/" + Test.externalGroupVersion.Version + "/prefix/namespaces/mynamespace/resource/myresource"},
		{"prefix", "resource", "", "myresource", "/api/" + Test.externalGroupVersion.Version + "/prefix/resource/myresource"},
		{"prefix", "resource", "mynamespace", "", "/api/" + Test.externalGroupVersion.Version + "/prefix/namespaces/mynamespace/resource"},
		{"prefix", "resource", "", "", "/api/" + Test.externalGroupVersion.Version + "/prefix/resource"},
		{"", "resource", "mynamespace", "myresource", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
	}
	for _, item := range testCases {
		actual := Test.ResourcePathWithPrefix(item.prefix, item.resource, item.namespace, item.name)
		require.Equal(t, actual, item.expected, "Expected: %s, got: %s for prefix: %s, resource: %s, namespace: %s and name: %s", item.expected, actual, item.prefix, item.resource, item.namespace, item.name)
	}

	TestGroup := Test
	TestGroup.externalGroupVersion.Group = "TestGroup"

	testGroupCases := []struct {
		prefix    string
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"prefix", "resource", "mynamespace", "myresource", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/prefix/namespaces/mynamespace/resource/myresource"},
		{"prefix", "resource", "", "myresource", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/prefix/resource/myresource"},
		{"prefix", "resource", "mynamespace", "", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/prefix/namespaces/mynamespace/resource"},
		{"prefix", "resource", "", "", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/prefix/resource"},
		{"", "resource", "mynamespace", "myresource", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
	}
	for _, item := range testGroupCases {
		actual := TestGroup.ResourcePathWithPrefix(item.prefix, item.resource, item.namespace, item.name)
		require.Equal(t, actual, item.expected, "Expected: %s, got: %s for prefix: %s, resource: %s, namespace: %s and name: %s", item.expected, actual, item.prefix, item.resource, item.namespace, item.name)
	}

}

func TestResourcePath(t *testing.T) {
	testCases := []struct {
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"resource", "mynamespace", "myresource", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
		{"resource", "", "myresource", "/api/" + Test.externalGroupVersion.Version + "/resource/myresource"},
		{"resource", "mynamespace", "", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource"},
		{"resource", "", "", "/api/" + Test.externalGroupVersion.Version + "/resource"},
	}
	for _, item := range testCases {
		actual := Test.ResourcePath(item.resource, item.namespace, item.name)
		require.Equal(t, actual, item.expected, "Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
	}

	TestGroup := Test
	TestGroup.externalGroupVersion.Group = "TestGroup"

	testGroupCases := []struct {
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"resource", "mynamespace", "myresource", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
		{"resource", "", "myresource", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/resource/myresource"},
		{"resource", "mynamespace", "", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource"},
		{"resource", "", "", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/resource"},
	}
	for _, item := range testGroupCases {
		actual := TestGroup.ResourcePath(item.resource, item.namespace, item.name)
		require.Equal(t, actual, item.expected, "Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
	}

}

func TestSubResourcePath(t *testing.T) {
	testCases := []struct {
		resource  string
		namespace string
		name      string
		sub       string
		expected  string
	}{
		{"resource", "mynamespace", "myresource", "subresource", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource/subresource"},
		{"resource", "mynamespace", "myresource", "", "/api/" + Test.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
	}
	for _, item := range testCases {
		actual := Test.SubResourcePath(item.resource, item.namespace, item.name, item.sub)
		require.Equal(t, actual, item.expected, "Expected: %s, got: %s for resource: %s, namespace: %s, name: %s and sub: %s", item.expected, actual, item.resource, item.namespace, item.name, item.sub)
	}

	TestGroup := Test
	TestGroup.externalGroupVersion.Group = "TestGroup"

	testGroupCases := []struct {
		resource  string
		namespace string
		name      string
		sub       string
		expected  string
	}{
		{"resource", "mynamespace", "myresource", "subresource", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource/subresource"},
		{"resource", "mynamespace", "myresource", "", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/namespaces/mynamespace/resource/myresource"},
	}
	for _, item := range testGroupCases {
		actual := TestGroup.SubResourcePath(item.resource, item.namespace, item.name, item.sub)
		require.Equal(t, actual, item.expected, "Expected: %s, got: %s for resource: %s, namespace: %s, name: %s and sub: %s", item.expected, actual, item.resource, item.namespace, item.name, item.sub)
	}

}

func TestSelfLink(t *testing.T) {
	testCases := []struct {
		resource string
		name     string
		expected string
	}{
		{"resource", "name", "/api/" + Test.externalGroupVersion.Version + "/resource/name"},
		{"resource", "", "/api/" + Test.externalGroupVersion.Version + "/resource"},
	}
	for _, item := range testCases {
		actual := Test.SelfLink(item.resource, item.name)
		require.Equal(t, actual, item.expected, "Expected: %s, got: %s for resource: %s and name: %s", item.expected, actual, item.resource, item.name)
	}

	TestGroup := Test
	TestGroup.externalGroupVersion.Group = "TestGroup"

	testGroupCases := []struct {
		resource string
		name     string
		expected string
	}{
		{"resource", "name", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/resource/name"},
		{"resource", "", "/apis/" + TestGroup.externalGroupVersion.Group + "/" + TestGroup.externalGroupVersion.Version + "/resource"},
	}
	for _, item := range testGroupCases {
		actual := TestGroup.SelfLink(item.resource, item.name)
		require.Equal(t, actual, item.expected, "Expected: %s, got: %s for resource: %s and name: %s", item.expected, actual, item.resource, item.name)
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
	require.NoError(t, err, "unexpected error: %v", err)

	typeMeta := metav1.TypeMeta{}
	err = json.Unmarshal(encoded, &typeMeta)
	require.NoError(t, err, "unexpected error: %v", err)
	require.Equal(t, typeMeta.Kind, "Status", "Kind is not set to \"Status\". Got %v", string(encoded))
	require.Equal(t, typeMeta.APIVersion, "v1", "APIVersion is not set to \"v1\". Got %v", string(encoded))

	decoded, err := runtime.Decode(v1Codec, encoded)
	require.NoError(t, err, "unexpected error: %v", err)
	if !reflect.DeepEqual(status, decoded) {
		t.Errorf("expected: %#v, got: %#v", status, decoded)
	}
}
