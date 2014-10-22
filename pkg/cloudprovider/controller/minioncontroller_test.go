/*
Copyright 2014 Google Inc. All rights reserved.

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

package controller

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider/fake"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func newMinionList(count int) api.MinionList {
	minions := []api.Minion{}
	for i := 0; i < count; i++ {
		minions = append(minions, api.Minion{
			ObjectMeta: api.ObjectMeta{
				Name: fmt.Sprintf("minion%d", i),
			},
		})
	}
	return api.MinionList{
		Items: minions,
	}
}

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func makeTestServer(t *testing.T, minionResponse serverResponse) (*httptest.Server, *util.FakeHandler) {
	fakeMinionHandler := util.FakeHandler{
		StatusCode:   minionResponse.statusCode,
		ResponseBody: util.EncodeJSON(minionResponse.obj),
	}
	mux := http.NewServeMux()
	mux.Handle("/api/"+testapi.Version()+"/minions", &fakeMinionHandler)
	mux.Handle("/api/"+testapi.Version()+"/minions/", &fakeMinionHandler)
	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		res.WriteHeader(http.StatusNotFound)
	})
	return httptest.NewServer(mux), &fakeMinionHandler
}

func TestSyncCreateMinion(t *testing.T) {
	testServer, minionHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newMinionList(1)})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	instances := []string{"minion0", "minion1"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	minionController := NewMinionController(&fakeCloud, ".*", nil, nil, client)
	if err := minionController.Sync(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	data := runtime.EncodeOrDie(testapi.Codec(), &api.Minion{ObjectMeta: api.ObjectMeta{Name: "minion1"}})
	minionHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/minions", "POST", &data)
}

func TestSyncDeleteMinion(t *testing.T) {
	testServer, minionHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newMinionList(2)})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	instances := []string{"minion0"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	minionController := NewMinionController(&fakeCloud, ".*", nil, nil, client)
	if err := minionController.Sync(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	minionHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/minions/minion1", "DELETE", nil)
}

func TestSyncMinionRegexp(t *testing.T) {
	testServer, minionHandler := makeTestServer(t,
		serverResponse{http.StatusOK, newMinionList(1)})
	defer testServer.Close()
	client := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	instances := []string{"minion0", "minion1", "node0"}
	fakeCloud := fake_cloud.FakeCloud{
		Machines: instances,
	}
	minionController := NewMinionController(&fakeCloud, "minion[0-9]+", nil, nil, client)
	if err := minionController.Sync(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Only minion1 is created.
	data := runtime.EncodeOrDie(testapi.Codec(), &api.Minion{ObjectMeta: api.ObjectMeta{Name: "minion1"}})
	minionHandler.ValidateRequest(t, "/api/"+testapi.Version()+"/minions", "POST", &data)
}
