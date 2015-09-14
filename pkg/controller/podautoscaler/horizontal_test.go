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

package podautoscaler

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/experimental"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"

	"github.com/golang/glog"
	"github.com/stretchr/testify/assert"
)

const (
	namespace    = api.NamespaceDefault
	rcName       = "app-rc"
	podNameLabel = "app"
	hpaName      = "foo"

	hpaListHandler   = "HpaList"
	scaleHandler     = "Scale"
	updateHpaHandler = "HpaUpdate"
)

type serverResponse struct {
	statusCode int
	obj        interface{}
}

type fakeMetricsClient struct {
	consumption metrics.ResourceConsumptionClient
}

type fakeResourceConsumptionClient struct {
	metrics map[api.ResourceName]experimental.ResourceConsumption
}

func (f *fakeMetricsClient) ResourceConsumption(namespace string) metrics.ResourceConsumptionClient {
	return f.consumption
}

func (f *fakeResourceConsumptionClient) Get(resource api.ResourceName, selector map[string]string) (*experimental.ResourceConsumption, error) {
	consumption, found := f.metrics[resource]
	if !found {
		return nil, fmt.Errorf("resource not found: %v", resource)
	}
	return &consumption, nil
}

func makeTestServer(t *testing.T, responses map[string]*serverResponse) (*httptest.Server, map[string]*util.FakeHandler) {

	handlers := map[string]*util.FakeHandler{}
	mux := http.NewServeMux()

	mkHandler := func(url string, response serverResponse) *util.FakeHandler {
		handler := util.FakeHandler{
			StatusCode:   response.statusCode,
			ResponseBody: runtime.EncodeOrDie(testapi.Experimental.Codec(), response.obj.(runtime.Object)),
		}
		mux.Handle(url, &handler)
		glog.Infof("Will handle %s", url)
		return &handler
	}

	if responses[hpaListHandler] != nil {
		handlers[hpaListHandler] = mkHandler("/experimental/v1/horizontalpodautoscalers", *responses[hpaListHandler])
	}

	if responses[scaleHandler] != nil {
		handlers[scaleHandler] = mkHandler(
			fmt.Sprintf("/experimental/v1/namespaces/%s/replicationcontrollers/%s/scale", namespace, rcName), *responses[scaleHandler])
	}

	if responses[updateHpaHandler] != nil {
		handlers[updateHpaHandler] = mkHandler(fmt.Sprintf("/experimental/v1/namespaces/%s/horizontalpodautoscalers/%s", namespace, hpaName),
			*responses[updateHpaHandler])
	}

	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		res.WriteHeader(http.StatusNotFound)
	})
	return httptest.NewServer(mux), handlers
}

func TestSyncEndpointsItemsPreserveNoSelector(t *testing.T) {

	hpaResponse := serverResponse{http.StatusOK, &experimental.HorizontalPodAutoscalerList{
		Items: []experimental.HorizontalPodAutoscaler{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      hpaName,
					Namespace: namespace,
				},
				Spec: experimental.HorizontalPodAutoscalerSpec{
					ScaleRef: &experimental.SubresourceReference{
						Kind:        "replicationController",
						Name:        rcName,
						Namespace:   namespace,
						Subresource: "scale",
					},
					MinCount: 1,
					MaxCount: 5,
					Target:   experimental.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.3")},
				},
			}}}}

	scaleResponse := serverResponse{http.StatusOK, &experimental.Scale{
		ObjectMeta: api.ObjectMeta{
			Name:      rcName,
			Namespace: namespace,
		},
		Spec: experimental.ScaleSpec{
			Replicas: 1,
		},
		Status: experimental.ScaleStatus{
			Replicas: 1,
			Selector: map[string]string{"name": podNameLabel},
		},
	}}

	status := experimental.HorizontalPodAutoscalerStatus{
		CurrentReplicas: 1,
		DesiredReplicas: 3,
	}
	updateHpaResponse := serverResponse{http.StatusOK, &experimental.HorizontalPodAutoscaler{

		ObjectMeta: api.ObjectMeta{
			Name:      hpaName,
			Namespace: namespace,
		},
		Spec: experimental.HorizontalPodAutoscalerSpec{
			ScaleRef: &experimental.SubresourceReference{
				Kind:        "replicationController",
				Name:        rcName,
				Namespace:   namespace,
				Subresource: "scale",
			},
			MinCount: 1,
			MaxCount: 5,
			Target:   experimental.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse("0.3")},
		},
		Status: &status,
	}}

	testServer, handlers := makeTestServer(t,
		map[string]*serverResponse{
			hpaListHandler:   &hpaResponse,
			scaleHandler:     &scaleResponse,
			updateHpaHandler: &updateHpaResponse,
		})

	defer testServer.Close()
	kubeClient := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Experimental.Version()})
	fakeRC := fakeResourceConsumptionClient{metrics: map[api.ResourceName]experimental.ResourceConsumption{
		api.ResourceCPU: {Resource: api.ResourceCPU, Quantity: resource.MustParse("650m")},
	}}
	fake := fakeMetricsClient{consumption: &fakeRC}

	hpaController := NewHorizontalController(kubeClient, &fake)

	err := hpaController.reconcileAutoscalers()
	if err != nil {
		t.Fatal("Failed to reconcile: %v", err)
	}
	for _, h := range handlers {
		h.ValidateRequestCount(t, 1)
	}
	obj, err := kubeClient.Codec.Decode([]byte(handlers[updateHpaHandler].RequestBody))
	if err != nil {
		t.Fatal("Failed to decode: %v %v", err)
	}
	hpa, _ := obj.(*experimental.HorizontalPodAutoscaler)

	assert.Equal(t, 3, hpa.Status.DesiredReplicas)
	assert.Equal(t, int64(650), hpa.Status.CurrentConsumption.Quantity.MilliValue())
	assert.NotNil(t, hpa.Status.LastScaleTimestamp)
}
