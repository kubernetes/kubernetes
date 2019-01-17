/*
Copyright 2018 The Kubernetes Authors.

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

package endpoints

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapitesting "k8s.io/apiserver/pkg/endpoints/testing"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
)

func TestPatch(t *testing.T) {
	storage := map[string]rest.Storage{}
	ID := "id"
	item := &genericapitesting.Simple{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ID,
			Namespace: "", // update should allow the client to send an empty namespace
			UID:       "uid",
		},
		Other: "bar",
	}
	simpleStorage := SimpleRESTStorage{item: *item}
	storage["simple"] = &simpleStorage
	selfLinker := &setTestSelfLinker{
		t:           t,
		expectedSet: "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/namespaces/default/simple/" + ID,
		name:        ID,
		namespace:   metav1.NamespaceDefault,
	}
	handler := handleLinker(storage, selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

	client := http.Client{}
	request, err := http.NewRequest("PATCH", server.URL+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/"+ID, bytes.NewReader([]byte(`{"labels":{"foo":"bar"}}`)))
	request.Header.Set("Content-Type", "application/merge-patch+json; charset=UTF-8")
	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	dump, _ := httputil.DumpResponse(response, true)
	t.Log(string(dump))

	if simpleStorage.updated == nil || simpleStorage.updated.Labels["foo"] != "bar" {
		t.Errorf("Unexpected update value %#v, expected %#v.", simpleStorage.updated, item)
	}
	if !selfLinker.called {
		t.Errorf("Never set self link")
	}
}

func TestForbiddenForceOnNonApply(t *testing.T) {
	storage := map[string]rest.Storage{}
	ID := "id"
	item := &genericapitesting.Simple{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ID,
			Namespace: "", // update should allow the client to send an empty namespace
			UID:       "uid",
		},
		Other: "bar",
	}
	simpleStorage := SimpleRESTStorage{item: *item}
	storage["simple"] = &simpleStorage
	selfLinker := &setTestSelfLinker{
		t:           t,
		expectedSet: "/" + prefix + "/" + testGroupVersion.Group + "/" + testGroupVersion.Version + "/namespaces/default/simple/" + ID,
		name:        ID,
		namespace:   metav1.NamespaceDefault,
	}
	handler := handleLinker(storage, selfLinker)
	server := httptest.NewServer(handler)
	defer server.Close()

	client := http.Client{}
	request, err := http.NewRequest("PATCH", server.URL+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/"+ID, bytes.NewReader([]byte(`{"labels":{"foo":"bar"}}`)))
	request.Header.Set("Content-Type", "application/merge-patch+json; charset=UTF-8")
	_, err = client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	request, err = http.NewRequest("PATCH", server.URL+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/"+ID+"?force=true", bytes.NewReader([]byte(`{"labels":{"foo":"bar"}}`)))
	request.Header.Set("Content-Type", "application/merge-patch+json; charset=UTF-8")
	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if response.StatusCode != http.StatusUnprocessableEntity {
		t.Errorf("Unexpected response %#v", response)
	}

	request, err = http.NewRequest("PATCH", server.URL+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/"+ID+"?force=false", bytes.NewReader([]byte(`{"labels":{"foo":"bar"}}`)))
	request.Header.Set("Content-Type", "application/merge-patch+json; charset=UTF-8")
	response, err = client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if response.StatusCode != http.StatusUnprocessableEntity {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestPatchRequiresMatchingName(t *testing.T) {
	storage := map[string]rest.Storage{}
	ID := "id"
	item := &genericapitesting.Simple{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ID,
			Namespace: "", // update should allow the client to send an empty namespace
			UID:       "uid",
		},
		Other: "bar",
	}
	simpleStorage := SimpleRESTStorage{item: *item}
	storage["simple"] = &simpleStorage
	handler := handle(storage)
	server := httptest.NewServer(handler)
	defer server.Close()

	client := http.Client{}
	request, err := http.NewRequest("PATCH", server.URL+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/"+ID, bytes.NewReader([]byte(`{"metadata":{"name":"idbar"}}`)))
	request.Header.Set("Content-Type", "application/merge-patch+json")
	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if response.StatusCode != http.StatusBadRequest {
		t.Errorf("Unexpected response %#v", response)
	}
}

func TestPatchApply(t *testing.T) {
	t.Skip("apply is being refactored")
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()
	storage := map[string]rest.Storage{}
	item := &genericapitesting.Simple{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "id",
			Namespace: "",
			UID:       "uid",
		},
		Other: "bar",
	}
	simpleStorage := SimpleRESTStorage{item: *item}
	storage["simple"] = &simpleStorage
	handler := handle(storage)
	server := httptest.NewServer(handler)
	defer server.Close()

	client := http.Client{}
	request, err := http.NewRequest(
		"PATCH",
		server.URL+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/id",
		bytes.NewReader([]byte(`{"metadata":{"name":"id"}, "labels": {"test": "yes"}}`)),
	)
	request.Header.Set("Content-Type", "application/apply-patch+yaml")
	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if response.StatusCode != http.StatusOK {
		t.Errorf("Unexpected response %#v", response)
	}
	if simpleStorage.updated.Labels["test"] != "yes" {
		t.Errorf(`Expected labels to have "test": "yes", found %q`, simpleStorage.updated.Labels["test"])
	}
	if simpleStorage.updated.Other != "bar" {
		t.Errorf(`Merge should have kept initial "bar" value for Other: %v`, simpleStorage.updated.Other)
	}
	if _, ok := simpleStorage.updated.ObjectMeta.ManagedFields["default"]; !ok {
		t.Errorf(`Expected managedFields field to be set, but is empty`)
	}
}

func TestApplyAddsGVK(t *testing.T) {
	t.Skip("apply is being refactored")
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()
	storage := map[string]rest.Storage{}
	item := &genericapitesting.Simple{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "id",
			Namespace: "",
			UID:       "uid",
		},
		Other: "bar",
	}
	simpleStorage := SimpleRESTStorage{item: *item}
	storage["simple"] = &simpleStorage
	handler := handle(storage)
	server := httptest.NewServer(handler)
	defer server.Close()

	client := http.Client{}
	request, err := http.NewRequest(
		"PATCH",
		server.URL+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/id",
		bytes.NewReader([]byte(`{"metadata":{"name":"id"}, "labels": {"test": "yes"}}`)),
	)
	request.Header.Set("Content-Type", "application/apply-patch+yaml")
	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if response.StatusCode != http.StatusOK {
		t.Errorf("Unexpected response %#v", response)
	}
	// TODO: Need to fix this
	expected := `{"apiVersion":"test.group/version","kind":"Simple","labels":{"test":"yes"},"metadata":{"name":"id"}}`
	if simpleStorage.updated.ObjectMeta.ManagedFields["default"].APIVersion != expected {
		t.Errorf(
			`Expected managedFields field to be %q, got %q`,
			expected,
			simpleStorage.updated.ObjectMeta.ManagedFields["default"].APIVersion,
		)
	}
}

func TestApplyCreatesWithManagedFields(t *testing.T) {
	t.Skip("apply is being refactored")
	defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ServerSideApply, true)()
	storage := map[string]rest.Storage{}
	simpleStorage := SimpleRESTStorage{}
	storage["simple"] = &simpleStorage
	handler := handle(storage)
	server := httptest.NewServer(handler)
	defer server.Close()

	client := http.Client{}
	request, err := http.NewRequest(
		"PATCH",
		server.URL+"/"+prefix+"/"+testGroupVersion.Group+"/"+testGroupVersion.Version+"/namespaces/default/simple/id",
		bytes.NewReader([]byte(`{"metadata":{"name":"id"}, "labels": {"test": "yes"}}`)),
	)
	request.Header.Set("Content-Type", "application/apply-patch+yaml")
	response, err := client.Do(request)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if response.StatusCode != http.StatusOK {
		t.Errorf("Unexpected response %#v", response)
	}
	// TODO: Need to fix this
	expected := `{"apiVersion":"test.group/version","kind":"Simple","labels":{"test":"yes"},"metadata":{"name":"id"}}`
	if simpleStorage.updated.ObjectMeta.ManagedFields["default"].APIVersion != expected {
		t.Errorf(
			`Expected managedFields field to be %q, got %q`,
			expected,
			simpleStorage.updated.ObjectMeta.ManagedFields["default"].APIVersion,
		)
	}
}
