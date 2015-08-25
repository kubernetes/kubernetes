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

package autoscalercontroller

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
)

const (
	namespace    = api.NamespaceDefault
	rcName       = "app-rc"
	podNameLabel = "app"
	podName      = "p1"
)

var target = expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.8")}

type serverResponse struct {
	statusCode int
	obj        interface{}
}

func makeTestServer(t *testing.T, hpaResponse serverResponse,
	scaleResponse serverResponse, podListResponse serverResponse,
	heapsterResponse serverResponse) (*httptest.Server, []*util.FakeHandler) {

	handlers := []*util.FakeHandler{}
	mux := http.NewServeMux()

	mkHandler := func(url string, response serverResponse) *util.FakeHandler {
		handler := util.FakeHandler{
			StatusCode:   response.statusCode,
			ResponseBody: runtime.EncodeOrDie(testapi.Codec(), response.obj.(runtime.Object)),
		}
		mux.Handle(url, &handler)
		glog.Infof("Will handle %s", url)
		return &handler
	}

	mkRawHandler := func(url string, response serverResponse) *util.FakeHandler {
		handler := util.FakeHandler{
			StatusCode:   response.statusCode,
			ResponseBody: *response.obj.(*string),
		}
		mux.Handle(url, &handler)
		glog.Infof("Will handle %s", url)
		return &handler
	}

	handlers = append(handlers, mkHandler("/experimental/v1/horizontalpodautoscalers", hpaResponse))
	handlers = append(handlers, mkHandler(
		fmt.Sprintf("/experimental/v1/namespaces/%s/replicationcontrollers/%s/scale", namespace, rcName), scaleResponse))
	handlers = append(handlers, mkHandler(fmt.Sprintf("/api/v1/namespaces/%s/pods", namespace), podListResponse))
	handlers = append(handlers, mkRawHandler(
		fmt.Sprintf("/api/v1/proxy/namespaces/kube-system/services/monitoring-heapster/api/v1/model/namespaces/%s/pod-list/%s/metrics/cpu-usage",
			namespace, podName), heapsterResponse))

	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		res.WriteHeader(http.StatusNotFound)
	})
	return httptest.NewServer(mux), handlers
}

func TestSyncEndpointsItemsPreserveNoSelector(t *testing.T) {

	hpaResponse := serverResponse{http.StatusOK, &expapi.HorizontalPodAutoscalerList{
		Items: []expapi.HorizontalPodAutoscaler{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: namespace,
				},
				Spec: expapi.HorizontalPodAutoscalerSpec{
					ScaleRef: &expapi.SubresourceReference{
						Kind:        "replicationController",
						Name:        rcName,
						Namespace:   namespace,
						Subresource: "scale",
					},
					MinCount: 1,
					MaxCount: 5,
					Target:   target,
				},
			}}}}

	scaleResponse := serverResponse{http.StatusOK, &expapi.Scale{
		ObjectMeta: api.ObjectMeta{
			Name:      "rcName",
			Namespace: namespace,
		},
		Spec: expapi.ScaleSpec{
			Replicas: 5,
		},
		Status: expapi.ScaleStatus{
			Replicas: 2,
			Selector: map[string]string{"name": podNameLabel},
		},
	}}

	podListResponse := serverResponse{http.StatusOK, &api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      podName,
					Namespace: namespace,
				},
			}}}}

	heapsterRawResponse := "UPADTE ME"
	heapsterResponse := serverResponse{http.StatusOK, &heapsterRawResponse}

	testServer, handlers := makeTestServer(t, hpaResponse, scaleResponse, podListResponse, heapsterResponse)
	defer testServer.Close()
	kubeClient := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})
	expClient := client.NewExperimentalOrDie(&client.Config{Host: testServer.URL, Version: testapi.Version()})

	hpaController := New(kubeClient, expClient)
	err := hpaController.reconcileAutoscalers()
	if err != nil {
		t.Fatal("Failed to reconcile %v", err)
	}
	for _, h := range handlers {
		h.ValidateRequestCount(t, 1)
	}
}
