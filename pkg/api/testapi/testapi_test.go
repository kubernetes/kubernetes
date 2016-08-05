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

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// TODO these tests don't add much value for testing things that have groups

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

var status = &unversioned.Status{
	Status:  unversioned.StatusFailure,
	Code:    200,
	Reason:  unversioned.StatusReasonUnknown,
	Message: "",
}

func TestV1EncodeDecodeStatus(t *testing.T) {
	v1Codec := Default.Codec()

	encoded, err := runtime.Encode(v1Codec, status)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	typeMeta := unversioned.TypeMeta{}
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
	typeMeta := unversioned.TypeMeta{}
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
