// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v1

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	restful "github.com/emicklei/go-restful"
	"k8s.io/heapster/metrics/api/v1/types"
	"k8s.io/heapster/metrics/core"
)

type metricReq struct {
	name   string
	keys   []core.HistoricalKey
	start  time.Time
	end    time.Time
	labels map[string]string

	aggregations []core.AggregationType
	bucketSize   time.Duration
}

type fakeHistoricalSource struct {
	metricNames map[core.HistoricalKey][]string

	nodes             []string
	namespaces        []string
	podsForNamespace  map[string][]string
	containersForNode map[string][]string

	metricRequests      []metricReq
	aggregationRequests []metricReq

	nowTime time.Time
}

func (src *fakeHistoricalSource) GetMetric(metricName string, metricKeys []core.HistoricalKey, start, end time.Time) (map[core.HistoricalKey][]core.TimestampedMetricValue, error) {
	return src.GetLabeledMetric(metricName, nil, metricKeys, start, end)
}

func (src *fakeHistoricalSource) GetLabeledMetric(metricName string, labels map[string]string, metricKeys []core.HistoricalKey, start, end time.Time) (map[core.HistoricalKey][]core.TimestampedMetricValue, error) {
	if metricName == "invalid" {
		return nil, fmt.Errorf("fake error fetching metrics")
	}

	src.metricRequests = append(src.metricRequests, metricReq{
		name:   metricName,
		keys:   metricKeys,
		labels: labels,
		start:  start,
		end:    end,
	})

	res := make(map[core.HistoricalKey][]core.TimestampedMetricValue, len(metricKeys))

	for _, key := range metricKeys {
		res[key] = []core.TimestampedMetricValue{
			{
				Timestamp: src.nowTime.Add(-10 * time.Second),
				MetricValue: core.MetricValue{
					ValueType:  core.ValueFloat,
					FloatValue: 33,
				},
			},
		}
	}

	return res, nil
}

func (src *fakeHistoricalSource) GetAggregation(metricName string, aggregations []core.AggregationType, metricKeys []core.HistoricalKey, start, end time.Time, bucketSize time.Duration) (map[core.HistoricalKey][]core.TimestampedAggregationValue, error) {
	return src.GetLabeledAggregation(metricName, nil, aggregations, metricKeys, start, end, bucketSize)
}

func (src *fakeHistoricalSource) GetLabeledAggregation(metricName string, labels map[string]string, aggregations []core.AggregationType, metricKeys []core.HistoricalKey, start, end time.Time, bucketSize time.Duration) (map[core.HistoricalKey][]core.TimestampedAggregationValue, error) {
	if metricName == "invalid" {
		return nil, fmt.Errorf("fake error fetching metrics")
	}

	src.metricRequests = append(src.aggregationRequests, metricReq{
		name:   metricName,
		keys:   metricKeys,
		start:  start,
		end:    end,
		labels: labels,

		aggregations: aggregations,
		bucketSize:   bucketSize,
	})

	res := make(map[core.HistoricalKey][]core.TimestampedAggregationValue, len(metricKeys))

	for _, key := range metricKeys {
		countVal := uint64(10)
		res[key] = []core.TimestampedAggregationValue{
			{
				Timestamp:  src.nowTime.Add(-10 * time.Second),
				BucketSize: 10 * time.Second,
				AggregationValue: core.AggregationValue{
					Count: &countVal,
				},
			},
		}
	}

	return res, nil
}

func (src *fakeHistoricalSource) GetMetricNames(metricKey core.HistoricalKey) ([]string, error) {
	if names, ok := src.metricNames[metricKey]; ok {
		return names, nil
	}

	return nil, fmt.Errorf("no such object %q", metricKey.String())
}

func (src *fakeHistoricalSource) GetNodes() ([]string, error) {
	return src.nodes, nil
}

func (src *fakeHistoricalSource) GetNamespaces() ([]string, error) {
	return src.namespaces, nil
}

func (src *fakeHistoricalSource) GetPodsFromNamespace(namespace string) ([]string, error) {
	if pods, ok := src.podsForNamespace[namespace]; ok {
		return pods, nil
	}

	return nil, fmt.Errorf("no such namespace %q", namespace)
}

func (src *fakeHistoricalSource) GetSystemContainersFromNode(node string) ([]string, error) {
	if conts, ok := src.containersForNode[node]; ok {
		return conts, nil
	}

	return nil, fmt.Errorf("no such node %q", node)
}

func prepApi() (*HistoricalApi, *fakeHistoricalSource) {
	histSrc := &fakeHistoricalSource{
		metricNames: make(map[core.HistoricalKey][]string),
	}

	api := &HistoricalApi{
		Api: &Api{
			historicalSource: histSrc,
		},
	}

	return api, histSrc
}

type fakeRespRecorder struct {
	headers http.Header
	status  int
	data    *bytes.Buffer
}

func (r *fakeRespRecorder) Header() http.Header {
	return r.headers
}

func (r *fakeRespRecorder) WriteHeader(status int) {
	r.status = status
}

func (r *fakeRespRecorder) Write(content []byte) (int, error) {
	return r.data.Write(content)
}

func TestAvailableMetrics(t *testing.T) {
	api, src := prepApi()

	src.metricNames = map[core.HistoricalKey][]string{
		core.HistoricalKey{
			ObjectType: core.MetricSetTypeCluster,
		}: {"cm1", "cm2"},

		core.HistoricalKey{
			ObjectType: core.MetricSetTypeNode,
			NodeName:   "somenode1",
		}: {"nm1", "nm2"},

		core.HistoricalKey{
			ObjectType:    core.MetricSetTypeNamespace,
			NamespaceName: "somens1",
		}: {"nsm1", "nsm2"},

		core.HistoricalKey{
			ObjectType:    core.MetricSetTypePod,
			NamespaceName: "somens1",
			PodName:       "somepod1",
		}: {"pm1", "pm2"},

		core.HistoricalKey{
			ObjectType:    core.MetricSetTypePodContainer,
			NamespaceName: "somens1",
			PodName:       "somepod1",
			ContainerName: "somecont1",
		}: {"pcm1", "pcm2"},

		core.HistoricalKey{
			ObjectType:    core.MetricSetTypeSystemContainer,
			NodeName:      "somenode1",
			ContainerName: "somecont1",
		}: {"ncm1", "ncm2"},
	}

	tests := []struct {
		name          string
		fun           func(request *restful.Request, response *restful.Response)
		pathParams    map[string]string
		expectedNames []string
	}{
		{
			name:          "cluster metrics",
			fun:           api.availableClusterMetrics,
			pathParams:    map[string]string{},
			expectedNames: []string{"cm1", "cm2"},
		},
		{
			name:          "node metrics",
			fun:           api.availableNodeMetrics,
			pathParams:    map[string]string{"node-name": "somenode1"},
			expectedNames: []string{"nm1", "nm2"},
		},
		{
			name:          "namespace metrics",
			fun:           api.availableNamespaceMetrics,
			pathParams:    map[string]string{"namespace-name": "somens1"},
			expectedNames: []string{"nsm1", "nsm2"},
		},
		{
			name: "pod metrics",
			fun:  api.availablePodMetrics,
			pathParams: map[string]string{
				"namespace-name": "somens1",
				"pod-name":       "somepod1",
			},
			expectedNames: []string{"pm1", "pm2"},
		},
		{
			name: "pod container metrics",
			fun:  api.availablePodContainerMetrics,
			pathParams: map[string]string{
				"namespace-name": "somens1",
				"pod-name":       "somepod1",
				"container-name": "somecont1",
			},
			expectedNames: []string{"pcm1", "pcm2"},
		},
		{
			name: "free container metrics",
			fun:  api.availableFreeContainerMetrics,
			pathParams: map[string]string{
				"node-name":      "somenode1",
				"container-name": "somecont1",
			},
			expectedNames: []string{"ncm1", "ncm2"},
		},
	}

	assert := assert.New(t)
	restful.DefaultResponseMimeType = restful.MIME_JSON

	for _, test := range tests {
		req := restful.NewRequest(&http.Request{})
		pathParams := req.PathParameters()
		for k, v := range test.pathParams {
			pathParams[k] = v
		}
		recorder := &fakeRespRecorder{
			data:    new(bytes.Buffer),
			headers: make(http.Header),
		}
		resp := restful.NewResponse(recorder)

		test.fun(req, resp)

		actualNames := []string{}
		if err := json.Unmarshal(recorder.data.Bytes(), &actualNames); err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		assert.Equal(http.StatusOK, recorder.status, "status should have been OK (200)")
		assert.Equal(test.expectedNames, actualNames, "should have gotten expected JSON")
	}

	for _, test := range tests {
		if len(test.pathParams) == 0 {
			// don't test tests with no parameters for invalid parameters
			continue
		}
		req := restful.NewRequest(&http.Request{})
		pathParams := req.PathParameters()
		for k := range test.pathParams {
			pathParams[k] = "some-other-value"
		}
		recorder := &fakeRespRecorder{
			data:    new(bytes.Buffer),
			headers: make(http.Header),
		}
		resp := restful.NewResponse(recorder)

		test.fun(req, resp)

		assert.Equal(http.StatusInternalServerError, recorder.status, "status should have been InternalServerError (500)")
	}
}

func TestListObjects(t *testing.T) {
	api, src := prepApi()

	src.nodes = []string{"node1", "node2"}
	src.namespaces = []string{"ns1", "ns2"}
	src.podsForNamespace = map[string][]string{
		"ns1": {"pod1", "pod2"},
	}
	src.containersForNode = map[string][]string{
		"node1": {"x/y/z", "a/b/c"},
	}

	tests := []struct {
		name          string
		fun           func(request *restful.Request, response *restful.Response)
		pathParams    map[string]string
		expectedNames []string
	}{
		{
			name:          "nodes",
			fun:           api.nodeList,
			expectedNames: []string{"node1", "node2"},
		},
		{
			name:          "namespaces",
			fun:           api.namespaceList,
			expectedNames: []string{"ns1", "ns2"},
		},
		{
			name:          "pods in namespace",
			fun:           api.namespacePodList,
			pathParams:    map[string]string{"namespace-name": "ns1"},
			expectedNames: []string{"pod1", "pod2"},
		},
		{
			name: "free containers on node",
			fun:  api.nodeSystemContainerList,
			pathParams: map[string]string{
				"node-name": "node1",
			},
			expectedNames: []string{"x/y/z", "a/b/c"},
		},
	}

	assert := assert.New(t)
	restful.DefaultResponseMimeType = restful.MIME_JSON

	for _, test := range tests {
		req := restful.NewRequest(&http.Request{})
		pathParams := req.PathParameters()
		for k, v := range test.pathParams {
			pathParams[k] = v
		}
		recorder := &fakeRespRecorder{
			data:    new(bytes.Buffer),
			headers: make(http.Header),
		}
		resp := restful.NewResponse(recorder)

		test.fun(req, resp)

		actualNames := []string{}
		if err := json.Unmarshal(recorder.data.Bytes(), &actualNames); err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		assert.Equal(http.StatusOK, recorder.status, "status should have been OK (200)")
		assert.Equal(test.expectedNames, actualNames, "should have gotten expected JSON")
	}

	for _, test := range tests {
		if len(test.pathParams) == 0 {
			// don't test tests with no parameters for invalid parameters
			continue
		}
		req := restful.NewRequest(&http.Request{})
		pathParams := req.PathParameters()
		for k := range test.pathParams {
			pathParams[k] = "some-other-value"
		}
		recorder := &fakeRespRecorder{
			data:    new(bytes.Buffer),
			headers: make(http.Header),
		}
		resp := restful.NewResponse(recorder)

		test.fun(req, resp)

		assert.Equal(http.StatusInternalServerError, recorder.status, "status should have been InternalServerError (500)")
	}
}

func TestFetchMetrics(t *testing.T) {
	api, src := prepApi()
	nowTime := time.Now().UTC().Truncate(time.Second)
	src.nowTime = nowTime
	nowFunc = func() time.Time { return nowTime }

	tests := []struct {
		test              string
		start             string
		end               string
		labels            string
		fun               func(*restful.Request, *restful.Response)
		pathParams        map[string]string
		expectedMetricReq metricReq
		expectedStatus    int
	}{
		{
			test: "cluster metrics",
			fun:  api.clusterMetrics,
			pathParams: map[string]string{
				"metric-name": "some-metric",
			},
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeCluster},
				},
			},
		},
		{
			test: "node metrics",
			fun:  api.nodeMetrics,
			pathParams: map[string]string{
				"metric-name": "some-metric",
				"node-name":   "node1",
			},
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeNode, NodeName: "node1"},
				},
			},
		},
		{
			test: "namespace metrics",
			fun:  api.namespaceMetrics,
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"namespace-name": "ns1",
			},
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeNamespace, NamespaceName: "ns1"},
				},
			},
		},
		{
			test: "pod name metrics",
			fun:  api.podMetrics,
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"namespace-name": "ns1",
				"pod-name":       "pod1",
			},
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod1"},
				},
			},
		},
		{
			test: "pod id metrics",
			fun:  api.podMetrics,
			pathParams: map[string]string{
				"metric-name": "some-metric",
				"pod-id":      "pod-1-id",
			},
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, PodId: "pod-1-id"},
				},
			},
		},
		{
			test: "pod name container metrics",
			fun:  api.podContainerMetrics,
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"namespace-name": "ns1",
				"pod-name":       "pod1",
				"container-name": "cont1",
			},
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePodContainer, NamespaceName: "ns1", PodName: "pod1", ContainerName: "cont1"},
				},
			},
		},
		{
			test: "pod id container metrics",
			fun:  api.podContainerMetrics,
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"pod-id":         "pod-1-id",
				"container-name": "cont1",
			},
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePodContainer, PodId: "pod-1-id", ContainerName: "cont1"},
				},
			},
		},
		{
			test: "system container metrics",
			fun:  api.freeContainerMetrics,
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"node-name":      "node1",
				"container-name": "cont1",
			},
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeSystemContainer, NodeName: "node1", ContainerName: "cont1"},
				},
			},
		},
		{
			test: "query with end",
			fun:  api.clusterMetrics,
			pathParams: map[string]string{
				"metric-name": "some-metric",
			},
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			end:   nowTime.Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeCluster},
				},
				start: nowTime.Add(-10 * time.Second),
				end:   nowTime,
			},
		},
		{
			test: "query with labels",
			fun:  api.clusterMetrics,
			pathParams: map[string]string{
				"metric-name": "some-metric",
			},
			start:  nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			labels: "somelbl:v1,otherlbl:v2.3:4",
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeCluster},
				},
				labels: map[string]string{"somelbl": "v1", "otherlbl": "v2.3:4"},
				start:  nowTime.Add(-10 * time.Second),
			},
		},
		{
			test:           "query with bad start",
			fun:            api.clusterMetrics,
			start:          "afdsfd",
			expectedStatus: http.StatusBadRequest,
		},
		{
			test:           "query with no start",
			fun:            api.clusterMetrics,
			expectedStatus: http.StatusBadRequest,
		},
		{
			test:  "query with error while fetching metrics",
			fun:   api.clusterMetrics,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name": "invalid",
			},
			expectedStatus: http.StatusInternalServerError,
		},
		{
			test:           "query with bad labels",
			fun:            api.clusterMetrics,
			start:          nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			labels:         "abc:",
			expectedStatus: http.StatusBadRequest,
		},
	}

	assert := assert.New(t)
	restful.DefaultResponseMimeType = restful.MIME_JSON

	// doesn't particularly correspond to the query -- we're just using it to
	// test conversion between internal types
	expectedNormalVals := types.MetricResult{
		LatestTimestamp: nowTime.Add(-10 * time.Second),
		Metrics: []types.MetricPoint{
			{
				Timestamp: nowTime.Add(-10 * time.Second),
				Value:     33,
			},
		},
	}

	for _, test := range tests {
		queryParams := make(url.Values)
		queryParams.Add("start", test.start)
		queryParams.Add("end", test.end)
		queryParams.Add("labels", test.labels)
		u := &url.URL{RawQuery: queryParams.Encode()}
		req := restful.NewRequest(&http.Request{URL: u})
		pathParams := req.PathParameters()
		for k, v := range test.pathParams {
			pathParams[k] = v
		}
		recorder := &fakeRespRecorder{
			data:    new(bytes.Buffer),
			headers: make(http.Header),
		}
		resp := restful.NewResponse(recorder)

		test.fun(req, resp)

		if test.expectedStatus != 0 {
			assert.Equal(test.expectedStatus, recorder.status, "for test %q: status should have been an error status", test.test)
		} else {
			if !assert.Equal(http.StatusOK, recorder.status, "for test %q: status should have been OK (200)", test.test) {
				continue
			}

			actualReq := src.metricRequests[len(src.metricRequests)-1]
			if test.expectedMetricReq.end.IsZero() {
				test.expectedMetricReq.end = nowTime
			}
			assert.Equal(test.expectedMetricReq, actualReq, "for test %q: expected a different metric request to have been placed", test.test)

			actualVals := types.MetricResult{}
			if err := json.Unmarshal(recorder.data.Bytes(), &actualVals); err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			assert.Equal(expectedNormalVals, actualVals, "for test %q: should have gotten expected JSON", test.test)
		}
	}

	listTests := []struct {
		test              string
		start             string
		end               string
		labels            string
		fun               func(*restful.Request, *restful.Response)
		pathParams        map[string]string
		expectedMetricReq metricReq
		expectedStatus    int
	}{
		{
			test:  "pod list by ids",
			fun:   api.podListMetrics,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name": "some-metric",
				"pod-id-list": "pod-id-1,pod-id-2,pod-id-3",
			},
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, PodId: "pod-id-1"},
					{ObjectType: core.MetricSetTypePod, PodId: "pod-id-2"},
					{ObjectType: core.MetricSetTypePod, PodId: "pod-id-3"},
				},
				start: nowTime.Add(-10 * time.Second),
			},
		},
		{
			test:  "pod list by names",
			fun:   api.podListMetrics,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"namespace-name": "ns1",
				"pod-list":       "pod1,pod2,pod3",
			},
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod1"},
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod2"},
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod3"},
				},
				start: nowTime.Add(-10 * time.Second),
			},
		},
		{
			test: "pod list with labels",
			fun:  api.podListMetrics,
			pathParams: map[string]string{
				"metric-name": "some-metric",
				"pod-id-list": "pod-id-1,pod-id-2,pod-id-3",
			},
			start:  nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			labels: "somelbl:v1,otherlbl:v2.3:4",
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, PodId: "pod-id-1"},
					{ObjectType: core.MetricSetTypePod, PodId: "pod-id-2"},
					{ObjectType: core.MetricSetTypePod, PodId: "pod-id-3"},
				},
				labels: map[string]string{"somelbl": "v1", "otherlbl": "v2.3:4"},
				start:  nowTime.Add(-10 * time.Second),
			},
		},
		{
			test:           "pod list with bad start time",
			fun:            api.podListMetrics,
			start:          "afdsfd",
			expectedStatus: http.StatusBadRequest,
		},
		{
			test:  "pod list with error fetching metrics",
			fun:   api.podListMetrics,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name": "invalid",
			},
			expectedStatus: http.StatusInternalServerError,
		},
		{
			test:           "pod list with no start",
			fun:            api.podListMetrics,
			expectedStatus: http.StatusBadRequest,
		},
		{
			test:           "query with bad labels",
			fun:            api.clusterMetrics,
			start:          nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			labels:         "abc:",
			expectedStatus: http.StatusBadRequest,
		},
	}

	expectedListVals := types.MetricResultList{
		Items: []types.MetricResult{
			expectedNormalVals,
			expectedNormalVals,
			expectedNormalVals,
		},
	}

	for _, test := range listTests {
		queryParams := make(url.Values)
		queryParams.Add("start", test.start)
		queryParams.Add("end", test.end)
		queryParams.Add("labels", test.labels)
		u := &url.URL{RawQuery: queryParams.Encode()}
		req := restful.NewRequest(&http.Request{URL: u})
		pathParams := req.PathParameters()
		for k, v := range test.pathParams {
			pathParams[k] = v
		}
		recorder := &fakeRespRecorder{
			data:    new(bytes.Buffer),
			headers: make(http.Header),
		}
		resp := restful.NewResponse(recorder)

		test.fun(req, resp)

		if test.expectedStatus != 0 {
			assert.Equal(test.expectedStatus, recorder.status, "for test %q: status should have been an error status", test.test)
		} else {
			if !assert.Equal(http.StatusOK, recorder.status, "for test %q: status should have been OK (200)", test.test) {
				continue
			}

			actualReq := src.metricRequests[len(src.metricRequests)-1]
			if test.expectedMetricReq.end.IsZero() {
				test.expectedMetricReq.end = nowTime
			}

			assert.Equal(test.expectedMetricReq, actualReq, "for test %q: expected a different metric request to have been placed", test.test)

			actualVals := types.MetricResultList{}
			if err := json.Unmarshal(recorder.data.Bytes(), &actualVals); err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			assert.Equal(expectedListVals, actualVals, "for test %q: should have gotten expected JSON", test.test)
		}
	}
}

func TestFetchAggregations(t *testing.T) {
	api, src := prepApi()
	nowTime := time.Now().UTC().Truncate(time.Second)
	src.nowTime = nowTime
	nowFunc = func() time.Time { return nowTime }

	tests := []struct {
		test              string
		bucketSize        string
		start             string
		end               string
		labels            string
		fun               func(*restful.Request, *restful.Response)
		pathParams        map[string]string
		expectedMetricReq metricReq
		expectedStatus    int
	}{
		{
			test:  "cluster aggregations",
			fun:   api.clusterAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":  "some-metric",
				"aggregations": "count,average",
			},
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeCluster},
				},
			},
		},
		{
			test:  "node aggregations",
			fun:   api.nodeAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":  "some-metric",
				"node-name":    "node1",
				"aggregations": "count,average",
			},
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeNode, NodeName: "node1"},
				},
			},
		},
		{
			test:  "namespace aggregations",
			fun:   api.namespaceAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"namespace-name": "ns1",
				"aggregations":   "count,average",
			},
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeNamespace, NamespaceName: "ns1"},
				},
			},
		},
		{
			test:  "pod name aggregations",
			fun:   api.podAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"namespace-name": "ns1",
				"pod-name":       "pod1",
				"aggregations":   "count,average",
			},
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod1"},
				},
			},
		},
		{
			test:  "pod id aggregations",
			fun:   api.podAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":  "some-metric",
				"pod-id":       "pod-1-id",
				"aggregations": "count,average",
			},
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, PodId: "pod-1-id"},
				},
			},
		},
		{
			test:  "pod name container aggregations",
			fun:   api.podContainerAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"namespace-name": "ns1",
				"pod-name":       "pod1",
				"container-name": "cont1",
				"aggregations":   "count,average",
			},
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePodContainer, NamespaceName: "ns1", PodName: "pod1", ContainerName: "cont1"},
				},
			},
		},
		{
			test:  "pod id container aggregations",
			fun:   api.podContainerAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"pod-id":         "pod-1-id",
				"container-name": "cont1",
				"aggregations":   "count,average",
			},
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePodContainer, PodId: "pod-1-id", ContainerName: "cont1"},
				},
			},
		},
		{
			test:  "system container aggregations",
			fun:   api.freeContainerAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"node-name":      "node1",
				"container-name": "cont1",
				"aggregations":   "count,average",
			},
			expectedMetricReq: metricReq{
				name:  "some-metric",
				start: nowTime.Add(-10 * time.Second),
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeSystemContainer, NodeName: "node1", ContainerName: "cont1"},
				},
			},
		},
		{
			test: "aggregations with end and bucket",
			fun:  api.clusterAggregations,
			pathParams: map[string]string{
				"metric-name":  "some-metric",
				"aggregations": "count,average",
			},
			start:      nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			end:        nowTime.Format(time.RFC3339),
			bucketSize: "20s",
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeCluster},
				},
				start:      nowTime.Add(-10 * time.Second),
				end:        nowTime,
				bucketSize: 20 * time.Second,
			},
		},
		{
			test: "aggregations with labels",
			fun:  api.clusterAggregations,
			pathParams: map[string]string{
				"metric-name":  "some-metric",
				"aggregations": "count,average",
			},
			labels: "somelbl:v1,otherlbl:v2.3:4",
			start:  nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypeCluster},
				},
				start:  nowTime.Add(-10 * time.Second),
				labels: map[string]string{"somelbl": "v1", "otherlbl": "v2.3:4"},
			},
		},
		{
			test:           "aggregations with bad start time",
			fun:            api.clusterAggregations,
			start:          "afdsfd",
			expectedStatus: http.StatusBadRequest,
		},
		{
			test:  "aggregations with fetch error",
			fun:   api.clusterAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":  "invalid",
				"aggregations": "count,average",
			},
			expectedStatus: http.StatusInternalServerError,
		},
		{
			test:           "aggregations with no start time",
			fun:            api.clusterAggregations,
			expectedStatus: http.StatusBadRequest,
		},
	}

	assert := assert.New(t)
	restful.DefaultResponseMimeType = restful.MIME_JSON

	// doesn't particularly correspond to the query -- we're just using it to
	// test conversion between internal types
	countVal := uint64(10)
	expectedNormalVals := types.MetricAggregationResult{
		BucketSize: 10 * time.Second,
		Buckets: []types.MetricAggregationBucket{
			{
				Timestamp: src.nowTime.Add(-10 * time.Second),
				Count:     &countVal,
			},
		},
	}

	aggList := []core.AggregationType{
		core.AggregationTypeCount,
		core.AggregationTypeAverage,
	}

	for _, test := range tests {
		queryParams := make(url.Values)
		queryParams.Add("start", test.start)
		queryParams.Add("end", test.end)
		queryParams.Add("bucket", test.bucketSize)
		queryParams.Add("labels", test.labels)
		u := &url.URL{RawQuery: queryParams.Encode()}
		req := restful.NewRequest(&http.Request{URL: u})
		pathParams := req.PathParameters()
		for k, v := range test.pathParams {
			pathParams[k] = v
		}
		recorder := &fakeRespRecorder{
			data:    new(bytes.Buffer),
			headers: make(http.Header),
		}
		resp := restful.NewResponse(recorder)

		test.fun(req, resp)

		if test.expectedStatus != 0 {
			assert.Equal(test.expectedStatus, recorder.status, "for test %q: status should have been an error status", test.test)
		} else {
			if !assert.Equal(http.StatusOK, recorder.status, "for test %q: status should have been OK (200)", test.test) {
				continue
			}

			actualReq := src.metricRequests[len(src.metricRequests)-1]
			if test.expectedMetricReq.end.IsZero() {
				test.expectedMetricReq.end = nowTime
			}

			test.expectedMetricReq.aggregations = aggList
			assert.Equal(test.expectedMetricReq, actualReq, "for test %q: expected a different metric request to have been placed", test.test)

			actualVals := types.MetricAggregationResult{}
			if err := json.Unmarshal(recorder.data.Bytes(), &actualVals); err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			assert.Equal(expectedNormalVals, actualVals, "for test %q: should have gotten expected JSON", test.test)
		}
	}

	listTests := []struct {
		test              string
		start             string
		labels            string
		end               string
		fun               func(*restful.Request, *restful.Response)
		pathParams        map[string]string
		expectedMetricReq metricReq
		expectedStatus    int
	}{
		{
			test:  "pod id list aggregations",
			fun:   api.podListAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":  "some-metric",
				"pod-id-list":  "pod-id-1,pod-id-2,pod-id-3",
				"aggregations": "count,average",
			},
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, PodId: "pod-id-1"},
					{ObjectType: core.MetricSetTypePod, PodId: "pod-id-2"},
					{ObjectType: core.MetricSetTypePod, PodId: "pod-id-3"},
				},
				start: nowTime.Add(-10 * time.Second),
			},
		},
		{
			test:  "pod name list aggregations",
			fun:   api.podListAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"namespace-name": "ns1",
				"pod-list":       "pod1,pod2,pod3",
				"aggregations":   "count,average",
			},
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod1"},
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod2"},
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod3"},
				},
				start: nowTime.Add(-10 * time.Second),
			},
		},
		{
			test:   "pod list aggregations with labels",
			fun:    api.podListAggregations,
			start:  nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			labels: "somelbl:v1,otherlbl:v2.3:4",
			pathParams: map[string]string{
				"metric-name":    "some-metric",
				"namespace-name": "ns1",
				"pod-list":       "pod1,pod2,pod3",
				"aggregations":   "count,average",
			},
			expectedMetricReq: metricReq{
				name: "some-metric",
				keys: []core.HistoricalKey{
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod1"},
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod2"},
					{ObjectType: core.MetricSetTypePod, NamespaceName: "ns1", PodName: "pod3"},
				},
				start:  nowTime.Add(-10 * time.Second),
				labels: map[string]string{"somelbl": "v1", "otherlbl": "v2.3:4"},
			},
		},
		{
			test:           "pod list aggregations with bad start time",
			fun:            api.podListAggregations,
			start:          "afdsfd",
			expectedStatus: http.StatusBadRequest,
		},
		{
			test:  "pod list aggregations with fetch error",
			fun:   api.podListAggregations,
			start: nowTime.Add(-10 * time.Second).Format(time.RFC3339),
			pathParams: map[string]string{
				"metric-name":  "invalid",
				"aggregations": "count,average",
			},
			expectedStatus: http.StatusInternalServerError,
		},
		{
			test:           "pod list aggregations with no start time",
			fun:            api.podListAggregations,
			expectedStatus: http.StatusBadRequest,
		},
	}

	expectedListVals := types.MetricAggregationResultList{
		Items: []types.MetricAggregationResult{
			expectedNormalVals,
			expectedNormalVals,
			expectedNormalVals,
		},
	}

	for _, test := range listTests {
		queryParams := make(url.Values)
		queryParams.Add("start", test.start)
		queryParams.Add("end", test.end)
		queryParams.Add("labels", test.labels)
		u := &url.URL{RawQuery: queryParams.Encode()}
		req := restful.NewRequest(&http.Request{URL: u})
		pathParams := req.PathParameters()
		for k, v := range test.pathParams {
			pathParams[k] = v
		}
		recorder := &fakeRespRecorder{
			data:    new(bytes.Buffer),
			headers: make(http.Header),
		}
		resp := restful.NewResponse(recorder)

		test.fun(req, resp)

		if test.expectedStatus != 0 {
			assert.Equal(test.expectedStatus, recorder.status, "for test %q: status should have been an error status", test.test)
		} else {
			if !assert.Equal(http.StatusOK, recorder.status, "for test %q: status should have been OK (200)", test.test) {
				continue
			}

			actualReq := src.metricRequests[len(src.metricRequests)-1]
			if test.expectedMetricReq.end.IsZero() {
				test.expectedMetricReq.end = nowTime
			}
			test.expectedMetricReq.aggregations = aggList
			assert.Equal(test.expectedMetricReq, actualReq, "for test %q: expected a different metric request to have been placed", test.test)

			actualVals := types.MetricAggregationResultList{}
			if err := json.Unmarshal(recorder.data.Bytes(), &actualVals); err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			assert.Equal(expectedListVals, actualVals, "for test %q: should have gotten expected JSON", test.test)
		}
	}
}

func TestGetBucketSize(t *testing.T) {
	tests := []struct {
		sizeParam        string
		expectedDuration time.Duration
		expectedError    bool
	}{
		{
			// empty duration
			sizeParam:        "",
			expectedDuration: 0,
		},
		{
			// no units
			sizeParam:     "1",
			expectedError: true,
		},
		{
			// unknown unit
			sizeParam:     "1g",
			expectedError: true,
		},
		{
			// invalid unit (looks like ms)
			sizeParam:     "10gs",
			expectedError: true,
		},
		{
			// NaN
			sizeParam:     "abch",
			expectedError: true,
		},
		{
			sizeParam:        "5ms",
			expectedDuration: 5 * time.Millisecond,
		},
		{
			sizeParam:        "10s",
			expectedDuration: 10 * time.Second,
		},
		{
			sizeParam:        "15m",
			expectedDuration: 15 * time.Minute,
		},
		{
			sizeParam:        "20h",
			expectedDuration: 20 * time.Hour,
		},
		{
			sizeParam:        "25d",
			expectedDuration: 600 * time.Hour,
		},
	}

	assert := assert.New(t)
	for _, test := range tests {
		u, err := url.Parse(fmt.Sprintf("/foo?bucket=%s", test.sizeParam))
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		req := restful.NewRequest(&http.Request{URL: u})
		res, err := getBucketSize(req)
		if test.expectedError {
			assert.Error(err, fmt.Sprintf("%q should have been an invalid value", test.sizeParam))
		} else if assert.NoError(err, fmt.Sprintf("%q should have been a valid value", test.sizeParam)) {
			assert.Equal(test.expectedDuration, res, fmt.Sprintf("%q should have given the correct duration", test.sizeParam))
		}
	}
}

func TestGetAggregations(t *testing.T) {
	assert := assert.New(t)

	validAggregations := "average,max"
	invalidAggregations := "max,non-existant"

	req := restful.NewRequest(&http.Request{})
	pathParams := req.PathParameters()
	pathParams["aggregations"] = validAggregations

	validRes, validErr := getAggregations(req)
	if assert.NoError(validErr, "expected valid list to not produce an error") {
		assert.Equal([]core.AggregationType{core.AggregationTypeAverage, core.AggregationTypeMaximum}, validRes, "expected valid list to be properly split and converted to AggregationType values")
	}

	pathParams["aggregations"] = invalidAggregations
	_, invalidErr := getAggregations(req)
	assert.Error(invalidErr, "expected list with unknown aggregations to produce an error")
}

func TestExportMetricValue(t *testing.T) {
	assert := assert.New(t)

	assert.Nil(exportMetricValue(nil), "a nil input value should yield a nil output value")

	intVal := &core.MetricValue{
		IntValue:  33,
		ValueType: core.ValueInt64,
	}
	outputIntVal := exportMetricValue(intVal)
	if assert.NotNil(outputIntVal, "a non-nil input should yield a non-nil output") && assert.NotNil(outputIntVal.IntValue, "an int-valued input should yield an output with the IntValue set") {
		assert.Equal(intVal.IntValue, *outputIntVal.IntValue, "the input int value should be the same as the output int value")
	}

	floatVal := &core.MetricValue{
		FloatValue: 66.0,
		ValueType:  core.ValueFloat,
	}
	outputFloatVal := exportMetricValue(floatVal)
	if assert.NotNil(outputFloatVal, "a non-nil input should yield a non-nil output") && assert.NotNil(outputFloatVal.FloatValue, "an float-valued input should yield an output with the FloatValue set") {
		assert.Equal(float64(floatVal.FloatValue), *outputFloatVal.FloatValue, "the input float value should be the same as the output float value (as a float64)")
	}
}

func TestExtractMetricValue(t *testing.T) {
	assert := assert.New(t)

	aggregationVal := &core.AggregationValue{
		Aggregations: map[core.AggregationType]core.MetricValue{
			core.AggregationTypeAverage: {
				FloatValue: 66.0,
				ValueType:  core.ValueFloat,
			},
		},
	}

	avgRes := extractMetricValue(aggregationVal, core.AggregationTypeAverage)
	if assert.NotNil(avgRes, "a present aggregation should yield a non-nil output") && assert.NotNil(avgRes.FloatValue, "the output float value should be set when the input aggregation value is of the float type") {
		assert.Equal(66.0, *avgRes.FloatValue, "the output float value should be the same as the aggregation's float value")
	}

	maxRes := extractMetricValue(aggregationVal, core.AggregationTypeMaximum)
	assert.Nil(maxRes, "a non-present aggregation should yield a nil output")
}

func TestExportTimestampedAggregationValue(t *testing.T) {
	assert := assert.New(t)
	require := require.New(t)
	nowTime := time.Now().UTC()

	count10 := uint64(10)
	count11 := uint64(11)
	values := []core.TimestampedAggregationValue{
		{
			Timestamp:  nowTime.Add(-20 * time.Second),
			BucketSize: 10 * time.Second,
			AggregationValue: core.AggregationValue{
				Count: &count10,
				Aggregations: map[core.AggregationType]core.MetricValue{
					core.AggregationTypeAverage: {
						FloatValue: 66.0,
						ValueType:  core.ValueFloat,
					},
					core.AggregationTypePercentile50: {
						IntValue:  44,
						ValueType: core.ValueInt64,
					},
				},
			},
		},
		{
			Timestamp:  nowTime.Add(-10 * time.Second),
			BucketSize: 10 * time.Second,
			AggregationValue: core.AggregationValue{
				Count: &count11,
				Aggregations: map[core.AggregationType]core.MetricValue{
					core.AggregationTypeAverage: {
						FloatValue: 88.0,
						ValueType:  core.ValueFloat,
					},
					core.AggregationTypePercentile50: {
						IntValue:  55,
						ValueType: core.ValueInt64,
					},
				},
			},
		},
	}

	res := exportTimestampedAggregationValue(values)

	assert.Equal(10*time.Second, res.BucketSize, "the output bucket size should be 10s")
	require.Equal(len(values), len(res.Buckets), "there should be an output bucket for every input value")

	for i, bucket := range res.Buckets {
		inputVal := values[i]

		assert.Equal(inputVal.Count, bucket.Count, "the output bucket should have the same sample count as the input value")

		if assert.NotNil(bucket.Average, "the output bucket should have the average set") {
			metricVal := inputVal.Aggregations[core.AggregationTypeAverage]
			assert.Equal(exportMetricValue(&metricVal), bucket.Average, "the output bucket should have an average value equal to that of the input value")
		}

		percVal, ok := bucket.Percentiles["50"]
		if assert.True(ok, "the output bucket should have the 50th percentile set") {
			metricVal := inputVal.Aggregations[core.AggregationTypePercentile50]
			assert.Equal(exportMetricValue(&metricVal), &percVal, "the output bucket should have a 50th-percentile value equal to that of the input value")
		}
	}
}

func TestGetLabels(t *testing.T) {
	assert := assert.New(t)

	tests := []struct {
		test        string
		input       string
		outputVal   map[string]string
		outputError bool
	}{
		{
			test:      "working labels",
			input:     "k1:v1,k2:v2.3:4+5",
			outputVal: map[string]string{"k1": "v1", "k2": "v2.3:4+5"},
		},
		{
			test:        "bad label (no separator)",
			input:       "k1,k2:v2",
			outputError: true,
		},
		{
			test:        "bad label (no key)",
			input:       "k1:v1,:v2",
			outputError: true,
		},
		{
			test:        "bad label (no value)",
			input:       "k1:v1,k1:",
			outputError: true,
		},
		{
			test:      "empty",
			input:     "",
			outputVal: nil,
		},
	}

	for _, test := range tests {
		queryParams := make(url.Values)
		queryParams.Add("labels", test.input)
		u := &url.URL{RawQuery: queryParams.Encode()}
		req := restful.NewRequest(&http.Request{URL: u})
		res, err := getLabels(req)
		if test.outputError && !assert.Error(err, "test %q should have yielded an error", test.test) {
			continue
		} else if !test.outputError && !assert.NoError(err, "test %q should not have yielded an error", test.test) {
			continue
		}

		assert.Equal(test.outputVal, res, "test %q should have output the correct label map", test.test)
	}
}
