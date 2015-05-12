/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package serviceaccount

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func makeTestServer(t *testing.T, namespace string, serviceAccountResponse serverResponse) (*httptest.Server, *util.FakeHandler) {
	fakeServiceAccountsHandler := util.FakeHandler{
		StatusCode:   serviceAccountResponse.statusCode,
		ResponseBody: runtime.EncodeOrDie(testapi.Codec(), serviceAccountResponse.obj.(runtime.Object)),
	}

	mux := http.NewServeMux()
	mux.Handle(testapi.ResourcePath("serviceAccounts", namespace, ""), &fakeServiceAccountsHandler)
	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		res.WriteHeader(http.StatusNotFound)
	})
	return httptest.NewServer(mux), &fakeServiceAccountsHandler
}

func TestServiceAccountCreation(t *testing.T) {
	ns := api.NamespaceDefault

	activeNS := &api.Namespace{
		ObjectMeta: api.ObjectMeta{Name: ns},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceActive,
		},
	}
	terminatingNS := &api.Namespace{
		ObjectMeta: api.ObjectMeta{Name: ns},
		Status: api.NamespaceStatus{
			Phase: api.NamespaceTerminating,
		},
	}
	serviceAccount := &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:            "default",
			Namespace:       ns,
			ResourceVersion: "1",
		},
	}

	testcases := map[string]struct {
		ExistingNamespace      *api.Namespace
		ExistingServiceAccount *api.ServiceAccount

		AddedNamespace        *api.Namespace
		UpdatedNamespace      *api.Namespace
		DeletedServiceAccount *api.ServiceAccount

		ExpectCreatedServiceAccount bool
	}{
		"new active namespace missing serviceaccount": {
			AddedNamespace:              activeNS,
			ExpectCreatedServiceAccount: true,
		},
		"new active namespace with serviceaccount": {
			ExistingServiceAccount:      serviceAccount,
			AddedNamespace:              activeNS,
			ExpectCreatedServiceAccount: false,
		},
		"new terminating namespace": {
			AddedNamespace:              terminatingNS,
			ExpectCreatedServiceAccount: false,
		},

		"updated active namespace missing serviceaccount": {
			UpdatedNamespace:            activeNS,
			ExpectCreatedServiceAccount: true,
		},
		"updated active namespace with serviceaccount": {
			ExistingServiceAccount:      serviceAccount,
			UpdatedNamespace:            activeNS,
			ExpectCreatedServiceAccount: false,
		},
		"updated terminating namespace": {
			UpdatedNamespace:            terminatingNS,
			ExpectCreatedServiceAccount: false,
		},

		"deleted serviceaccount without namespace": {
			DeletedServiceAccount:       serviceAccount,
			ExpectCreatedServiceAccount: false,
		},
		"deleted serviceaccount with active namespace": {
			ExistingNamespace:           activeNS,
			DeletedServiceAccount:       serviceAccount,
			ExpectCreatedServiceAccount: true,
		},
		"deleted serviceaccount with terminating namespace": {
			ExistingNamespace:           terminatingNS,
			DeletedServiceAccount:       serviceAccount,
			ExpectCreatedServiceAccount: false,
		},
	}

	for k, tc := range testcases {

		testServer, handler := makeTestServer(t, ns, serverResponse{http.StatusOK, serviceAccount})
		client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
		controller := NewServiceAccountsController(client, DefaultServiceAccountControllerOptions())

		if tc.ExistingNamespace != nil {
			controller.namespaces.Add(tc.ExistingNamespace)
		}
		if tc.ExistingServiceAccount != nil {
			controller.serviceAccounts.Add(tc.ExistingServiceAccount)
		}

		if tc.AddedNamespace != nil {
			controller.namespaces.Add(tc.AddedNamespace)
			controller.namespaceAdded(tc.AddedNamespace)
		}
		if tc.UpdatedNamespace != nil {
			controller.namespaces.Add(tc.UpdatedNamespace)
			controller.namespaceUpdated(nil, tc.UpdatedNamespace)
		}
		if tc.DeletedServiceAccount != nil {
			controller.serviceAccountDeleted(tc.DeletedServiceAccount)
		}

		if tc.ExpectCreatedServiceAccount {
			if !handler.ValidateRequestCount(t, 1) {
				t.Errorf("%s: Expected a single creation call", k)
			}
		} else {
			if !handler.ValidateRequestCount(t, 0) {
				t.Errorf("%s: Expected no creation calls", k)
			}
		}

		testServer.Close()
	}
}
