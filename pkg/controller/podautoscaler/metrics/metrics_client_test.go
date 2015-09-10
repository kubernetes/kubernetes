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

package metrics

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/testapi"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"

	heapster "k8s.io/heapster/api/v1/types"

	"github.com/golang/glog"
	"github.com/stretchr/testify/assert"
)

const (
	namespace          = "test-namespace"
	podName            = "pod1"
	podListHandler     = "podlisthandler"
	heapsterCpuHandler = "heapstercpuhandler"
	heapsterMemHandler = "heapstermemhandler"
	cpu                = 650
	memory             = 20000000
)

type serverResponse struct {
	statusCode int
	obj        interface{}
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

	mkRawHandler := func(url string, response serverResponse) *util.FakeHandler {
		handler := util.FakeHandler{
			StatusCode:   response.statusCode,
			ResponseBody: *response.obj.(*string),
		}
		mux.Handle(url, &handler)
		glog.Infof("Will handle %s", url)
		return &handler
	}

	if responses[podListHandler] != nil {
		handlers[podListHandler] = mkHandler(fmt.Sprintf("/api/v1/namespaces/%s/pods", namespace), *responses[podListHandler])
	}

	if responses[heapsterCpuHandler] != nil {
		handlers[heapsterCpuHandler] = mkRawHandler(
			fmt.Sprintf("/api/v1/proxy/namespaces/kube-system/services/monitoring-heapster/api/v1/model/namespaces/%s/pod-list/%s/metrics/cpu-usage",
				namespace, podName), *responses[heapsterCpuHandler])
	}

	if responses[heapsterMemHandler] != nil {
		handlers[heapsterMemHandler] = mkRawHandler(
			fmt.Sprintf("/api/v1/proxy/namespaces/kube-system/services/monitoring-heapster/api/v1/model/namespaces/%s/pod-list/%s/metrics/memory-usage",
				namespace, podName), *responses[heapsterMemHandler])
	}

	mux.HandleFunc("/", func(res http.ResponseWriter, req *http.Request) {
		t.Errorf("unexpected request: %v", req.RequestURI)
		res.WriteHeader(http.StatusNotFound)
	})
	return httptest.NewServer(mux), handlers
}

func TestHeapsterResourceConsumptionGet(t *testing.T) {
	podListResponse := serverResponse{http.StatusOK, &api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      podName,
					Namespace: namespace,
				},
			}}}}

	timestamp := time.Now()
	metricsCpu := heapster.MetricResultList{
		Items: []heapster.MetricResult{{
			Metrics:         []heapster.MetricPoint{{timestamp, cpu}},
			LatestTimestamp: timestamp,
		}}}
	heapsterRawCpuResponse, _ := json.Marshal(&metricsCpu)
	heapsterStrCpuResponse := string(heapsterRawCpuResponse)
	heapsterCpuResponse := serverResponse{http.StatusOK, &heapsterStrCpuResponse}

	metricsMem := heapster.MetricResultList{
		Items: []heapster.MetricResult{{
			Metrics:         []heapster.MetricPoint{{timestamp, memory}},
			LatestTimestamp: timestamp,
		}}}
	heapsterRawMemResponse, _ := json.Marshal(&metricsMem)
	heapsterStrMemResponse := string(heapsterRawMemResponse)
	heapsterMemResponse := serverResponse{http.StatusOK, &heapsterStrMemResponse}

	testServer, _ := makeTestServer(t,
		map[string]*serverResponse{
			heapsterCpuHandler: &heapsterCpuResponse,
			heapsterMemHandler: &heapsterMemResponse,
			podListHandler:     &podListResponse,
		})

	defer testServer.Close()
	kubeClient := client.NewOrDie(&client.Config{Host: testServer.URL, Version: testapi.Experimental.Version()})

	metricsClient := NewHeapsterMetricsClient(kubeClient)

	val, err := metricsClient.ResourceConsumption(namespace).Get(api.ResourceCPU, map[string]string{"app": "test"})
	if err != nil {
		t.Fatalf("Error while getting consumption: %v", err)
	}
	assert.Equal(t, int64(cpu), val.Quantity.MilliValue())

	val, err = metricsClient.ResourceConsumption(namespace).Get(api.ResourceMemory, map[string]string{"app": "test"})
	if err != nil {
		t.Fatalf("Error while getting consumption: %v", err)
	}
	assert.Equal(t, int64(memory), val.Quantity.Value())
}
