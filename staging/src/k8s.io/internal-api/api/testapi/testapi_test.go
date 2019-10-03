/*
Copyright 2015 The Kubernetes Authors.

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
	"encoding/json"
	"reflect"
	"testing"

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
		{"prefix", "resource", "mynamespace", "myresource", "/api/" + Default.GroupVersion().Version + "/prefix/namespaces/mynamespace/resource/myresource"},
		{"prefix", "resource", "", "myresource", "/api/" + Default.GroupVersion().Version + "/prefix/resource/myresource"},
		{"prefix", "resource", "mynamespace", "", "/api/" + Default.GroupVersion().Version + "/prefix/namespaces/mynamespace/resource"},
		{"prefix", "resource", "", "", "/api/" + Default.GroupVersion().Version + "/prefix/resource"},
		{"", "resource", "mynamespace", "myresource", "/api/" + Default.GroupVersion().Version + "/namespaces/mynamespace/resource/myresource"},
	}
	for _, item := range testCases {
		if actual := Default.ResourcePathWithPrefix(item.prefix, item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for prefix: %s, resource: %s, namespace: %s and name: %s", item.expected, actual, item.prefix, item.resource, item.namespace, item.name)
		}
	}

	testGroupCases := []struct {
		prefix    string
		resource  string
		namespace string
		name      string
		expected  string
	}{
		{"prefix", "resource", "mynamespace", "myresource", "/apis/" + Admission.GroupVersion().Group + "/" + Admission.GroupVersion().Version + "/prefix/namespaces/mynamespace/resource/myresource"},
		{"prefix", "resource", "", "myresource", "/apis/" + Admission.GroupVersion().Group + "/" + Admission.GroupVersion().Version + "/prefix/resource/myresource"},
		{"prefix", "resource", "mynamespace", "", "/apis/" + Admission.GroupVersion().Group + "/" + Admission.GroupVersion().Version + "/prefix/namespaces/mynamespace/resource"},
		{"prefix", "resource", "", "", "/apis/" + Admission.GroupVersion().Group + "/" + Admission.GroupVersion().Version + "/prefix/resource"},
		{"", "resource", "mynamespace", "myresource", "/apis/" + Admission.GroupVersion().Group + "/" + Admission.GroupVersion().Version + "/namespaces/mynamespace/resource/myresource"},
	}
	for _, item := range testGroupCases {
		if actual := Admission.ResourcePathWithPrefix(item.prefix, item.resource, item.namespace, item.name); actual != item.expected {
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
		{"resource", "mynamespace", "myresource", "/api/" + Default.GroupVersion().Version + "/namespaces/mynamespace/resource/myresource"},
		{"resource", "", "myresource", "/api/" + Default.GroupVersion().Version + "/resource/myresource"},
		{"resource", "mynamespace", "", "/api/" + Default.GroupVersion().Version + "/namespaces/mynamespace/resource"},
		{"resource", "", "", "/api/" + Default.GroupVersion().Version + "/resource"},
	}
	for _, item := range testCases {
		if actual := Default.ResourcePath(item.resource, item.namespace, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
		}
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
		{"resource", "mynamespace", "myresource", "mysub", "/api/" + Default.GroupVersion().Version + "/namespaces/mynamespace/resource/myresource/mysub"},
		{"resource", "", "myresource", "mysub", "/api/" + Default.GroupVersion().Version + "/resource/myresource/mysub"},
		{"resource", "mynamespace", "", "mysub", "/api/" + Default.GroupVersion().Version + "/namespaces/mynamespace/resource/mysub"},
		{"resource", "", "", "mysub", "/api/" + Default.GroupVersion().Version + "/resource/mysub"},
	}
	for _, item := range testCases {
		if actual := Default.SubResourcePath(item.resource, item.namespace, item.name, item.sub); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for resource: %s, namespace: %s and name: %s", item.expected, actual, item.resource, item.namespace, item.name)
		}
	}
}

var status = &metav1.Status{
	Status:  metav1.StatusFailure,
	Code:    200,
	Reason:  metav1.StatusReasonUnknown,
	Message: "",
}

func TestSelfLink(t *testing.T) {
	testCases := []struct {
		resource string
		name     string
		expected string
	}{
		{"resource", "name", "/api/" + Default.GroupVersion().Version + "/resource/name"},
		{"resource", "", "/api/" + Default.GroupVersion().Version + "/resource"},
	}
	for _, item := range testCases {
		if actual := Default.SelfLink(item.resource, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for resource: %s and name: %s", item.expected, actual, item.resource, item.name)
		}
	}

	testGroupCases := []struct {
		resource string
		name     string
		expected string
	}{
		{"resource", "name", "/apis/" + Admission.GroupVersion().Group + "/" + Admission.GroupVersion().Version + "/resource/name"},
		{"resource", "", "/apis/" + Admission.GroupVersion().Group + "/" + Admission.GroupVersion().Version + "/resource"},
	}
	for _, item := range testGroupCases {
		if actual := Admission.SelfLink(item.resource, item.name); actual != item.expected {
			t.Errorf("Expected: %s, got: %s for resource: %s and name: %s", item.expected, actual, item.resource, item.name)
		}
	}
}

func TestV1EncodeDecodeStatus(t *testing.T) {
	v1Codec := Default.Codec()

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

func testEncodeDecodeStatus(t *testing.T, codec runtime.Codec) {
	encoded, err := runtime.Encode(codec, status)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	typeMeta := metav1.TypeMeta{}
	if err := json.Unmarshal(encoded, &typeMeta); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if typeMeta.Kind != "Status" {
		t.Errorf("Kind is not set to \"Status\". Got %s", encoded)
	}
	if typeMeta.APIVersion != "v1" {
		t.Errorf("APIVersion is not set to \"\". Got %s", encoded)
	}
	decoded, err := runtime.Decode(codec, encoded)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(status, decoded) {
		t.Errorf("expected: %v, got: %v", status, decoded)
	}
}

func TestAutoscalingEncodeDecodeStatus(t *testing.T) {
	testEncodeDecodeStatus(t, Autoscaling.Codec())
}

func TestBatchEncodeDecodeStatus(t *testing.T) {
	testEncodeDecodeStatus(t, Batch.Codec())
}

func TestExperimentalEncodeDecodeStatus(t *testing.T) {
	testEncodeDecodeStatus(t, Extensions.Codec())
}
