/*
Copyright 2021 The Kubernetes Authors.

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
package resource

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type fakeRESTClient struct {
	name string
	RESTClient
}

func TestClientPool(t *testing.T) {
	foov1GroupVersion := schema.GroupVersion{
		Group:   "foo",
		Version: "v1",
	}
	foov1beta1GroupVersion := schema.GroupVersion{
		Group:   "foo",
		Version: "v1beta1",
	}

	metav1Client := &fakeRESTClient{name: "metav1"}
	foov1Client := &fakeRESTClient{name: "foov1"}
	foov1beta1Client := &fakeRESTClient{name: "foov1"}
	clients := []RESTClient{
		metav1Client,
		foov1Client,
		foov1beta1Client,
		nil,
	}
	clientFuncCalls := make([]schema.GroupVersion, 0)
	clientFunc := func(version schema.GroupVersion) (RESTClient, error) {
		client := clients[len(clientFuncCalls)]
		clientFuncCalls = append(clientFuncCalls, version)

		if client == nil {
			return nil, errors.New("not found")
		}

		return client, nil
	}

	cp := newClientPool(clientFunc)
	assertClientPoolGet(t, cp, metav1.SchemeGroupVersion, metav1Client, nil)
	assertClientPoolGet(t, cp, metav1.SchemeGroupVersion, metav1Client, nil)
	assertClientPoolGet(t, cp, foov1GroupVersion, foov1Client, nil)
	assertClientPoolGet(t, cp, foov1GroupVersion, foov1Client, nil)
	assertClientPoolGet(t, cp, foov1beta1GroupVersion, foov1Client, nil)
	assertClientPoolGet(t, cp, foov1beta1GroupVersion, foov1beta1Client, nil)

	assertClientPoolGet(t, cp, foov1GroupVersion, foov1Client, nil)
	assertClientPoolGet(t, cp, foov1beta1GroupVersion, foov1Client, nil)
	assertClientPoolGet(t, cp, metav1.SchemeGroupVersion, metav1Client, nil)

	assertClientPoolGet(t, cp, schema.GroupVersion{}, nil, errors.New("not found"))

	wantCalls := []schema.GroupVersion{
		metav1.SchemeGroupVersion,
		foov1GroupVersion,
		foov1beta1GroupVersion,
		{},
	}
	if diff := cmp.Diff(clientFuncCalls, wantCalls); diff != "" {
		t.Fatalf("unexpected calls to client func: -got, +want:\n %s", diff)
	}
}

func assertClientPoolGet(
	t *testing.T,
	cp *clientPool,
	key schema.GroupVersion,
	wantClient *fakeRESTClient,
	wantError error,
) {
	t.Helper()

	client, err := cp.get(key)
	if err != nil {
		if wantError == nil || wantError.Error() != err.Error() {
			t.Fatalf("want error %q, got error %q", wantError, err)
		}
	} else {
		if wantError != nil {
			t.Fatalf("want error %q, got error <nil>", wantError)
		}
	}
	if client == nil {
		if wantClient != nil {
			t.Fatalf("want client %s, got client <nil>", wantClient.name)
		}
	} else {
		gotClient := client.(*fakeRESTClient)
		if gotClient.name != wantClient.name {
			t.Fatalf("want client %s, got client %s", wantClient.name, gotClient.name)
		}
	}
}
