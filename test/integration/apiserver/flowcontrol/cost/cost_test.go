/*
Copyright The Kubernetes Authors.

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

package cost

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// BenchmarkRequestCost measures per-request heap allocations (B/op and allocs/op)
// and wire bytes for kube-apiserver operations using the 10KB exemplar pod
// across request methods, options, and wire formats.
func BenchmarkRequestCost(b *testing.B) {
	featuregatetesting.SetFeatureGatesDuringTest(b, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CBORServingAndStorage:    true,
		features.DetectCacheInconsistency: false,
		features.WatchList:                true,
	})

	const (
		podPath  = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods/pod-000"
		podsPath = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods"
	)

	scenarios := []requestScenario{
		{
			method:             "GET",
			options:            `rv=""`,
			httpMethod:         http.MethodGet,
			path:               podPath,
			seedPods:           1,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "GET",
			options:            "rv=0",
			httpMethod:         http.MethodGet,
			path:               podPath,
			query:              "resourceVersion=0",
			seedPods:           1,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            `N=100 rv=""`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			seedPods:           100,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "N=100 rv=0",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=0",
			seedPods:           100,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "N=100 rv=Exact",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=LATEST_RV&resourceVersionMatch=Exact",
			seedPods:           100,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "N=100 limit=100",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "limit=100",
			seedPods:           100,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "WATCH",
			options:            `N=0 rv=""`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "N=1 fieldSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&fieldSelector=metadata.name=pod-000&timeoutSeconds=1",
			seedPods:           1,
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `N=100 rv=""`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&timeoutSeconds=1",
			seedPods:           100,
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "N=100 rv=0",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&timeoutSeconds=1",
			seedPods:           100,
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "N=100 rv=LATEST_RV",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=LATEST_RV&timeoutSeconds=1",
			seedPods:           100,
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "N=100 rv=FIRST_RV",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=FIRST_RV&timeoutSeconds=1",
			seedPods:           100,
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `N=100 rv="" sendInitialEvents=false`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=false&resourceVersionMatch=NotOlderThan&timeoutSeconds=1",
			seedPods:           100,
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "N=100 rv=0 sendInitialEvents=false",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=false&resourceVersionMatch=NotOlderThan&timeoutSeconds=1",
			seedPods:           100,
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `N=100 rv="" WatchList`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&timeoutSeconds=1",
			seedPods:           100,
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "N=100 rv=0 WatchList",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&timeoutSeconds=1",
			seedPods:           100,
			acceptContentTypes: watchFormats,
		},
		{
			method:             "POST",
			options:            "create",
			httpMethod:         http.MethodPost,
			path:               podsPath,
			query:              "fieldManager=bench",
			acceptContentTypes: writeFormats,
			body:               staticPodBody(b, "pod-new"),
		},
		{
			method:             "POST",
			options:            "create (dryRun)",
			httpMethod:         http.MethodPost,
			path:               podsPath,
			query:              "dryRun=All&fieldManager=bench",
			acceptContentTypes: writeFormats,
			body:               staticPodBody(b, "pod-new"),
		},
		{
			method:             "POST",
			options:            "binding",
			httpMethod:         http.MethodPost,
			path:               podPath + "/binding",
			query:              "fieldManager=bench",
			seedPods:           1,
			acceptContentTypes: writeFormats,
			body: staticObjectBody(b, &v1.Binding{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:   metav1.NamespaceDefault,
					Name:        "pod-000",
					Annotations: map[string]string{"bench.k8s.io/patch": "true"},
				},
				Target: v1.ObjectReference{
					Kind: "Node",
					Name: "kind-worker6",
				},
			}),
		},
		{
			method:             "PUT",
			options:            "update",
			httpMethod:         http.MethodPut,
			path:               podPath,
			query:              "fieldManager=bench",
			seedPods:           1,
			acceptContentTypes: writeFormats,
			body: func(mediaType string, pod *v1.Pod) []byte {
				updated := pod.DeepCopy()
				if updated.Annotations == nil {
					updated.Annotations = map[string]string{}
				}
				updated.Annotations["bench.k8s.io/patch"] = "true"
				return encodeObject(b, mediaType, updated)
			},
		},
		{
			method:             "PUT",
			options:            "status update",
			httpMethod:         http.MethodPut,
			path:               podPath + "/status",
			query:              "fieldManager=kubelet",
			seedPods:           1,
			acceptContentTypes: writeFormats,
			body: func(mediaType string, pod *v1.Pod) []byte {
				updated := pod.DeepCopy()
				updated.Status.Message = "updated-status"
				return encodeObject(b, mediaType, updated)
			},
		},
		{
			method:             "PATCH",
			options:            "strategic-merge",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench",
			seedPods:           1,
			requestContentType: string(types.StrategicMergePatchType),
			acceptContentTypes: writeFormats,
			body:               staticBytesBody(`{"metadata":{"annotations":{"bench.k8s.io/patch":"true"}}}`),
		},
		{
			method:             "PATCH",
			options:            "merge-patch",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench",
			seedPods:           1,
			requestContentType: string(types.MergePatchType),
			acceptContentTypes: writeFormats,
			body:               staticBytesBody(`{"metadata":{"annotations":{"bench.k8s.io/patch":"true"}}}`),
		},
		{
			method:             "PATCH",
			options:            "json-patch",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench",
			seedPods:           1,
			requestContentType: string(types.JSONPatchType),
			acceptContentTypes: writeFormats,
			body:               staticBytesBody(`[{"op":"add","path":"/metadata/annotations/bench.k8s.io~1patch","value":"true"}]`),
		},
		{
			method:             "PATCH",
			options:            "apply",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench&force=true",
			seedPods:           1,
			requestContentType: string(types.ApplyYAMLPatchType),
			acceptContentTypes: writeFormats,
			body:               staticBytesBody(`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"pod-000","annotations":{"bench.k8s.io/patch":"true"}}}`),
		},
		{
			method:             "PATCH",
			options:            "status strategic-merge",
			httpMethod:         http.MethodPatch,
			path:               podPath + "/status",
			query:              "fieldManager=kubelet",
			seedPods:           1,
			requestContentType: string(types.StrategicMergePatchType),
			acceptContentTypes: writeFormats,
			body:               staticBytesBody(`{"status":{"message":"updated-status"}}`),
		},
		{
			method:             "DELETE",
			options:            "gracePeriod=0",
			httpMethod:         http.MethodDelete,
			path:               podPath,
			query:              "gracePeriodSeconds=0",
			seedPods:           1,
			acceptContentTypes: writeFormats,
		},
		{
			method:             "DELETECOLLECTION",
			options:            "N=100",
			httpMethod:         http.MethodDelete,
			path:               podsPath,
			query:              "gracePeriodSeconds=0",
			seedPods:           100,
			acceptContentTypes: writeFormats,
		},
	}

	for _, sc := range scenarios {
		b.Run(fmt.Sprintf("Method=%s/Options=%s", sc.method, sc.options), func(b *testing.B) {
			benchmarkScenario(b, sc)
		})
	}
}

type requestScenario struct {
	method             string
	options            string
	httpMethod         string
	path               string
	query              string
	seedPods           int
	requestContentType string
	acceptContentTypes []contentType
	body               func(mediaType string, pod *v1.Pod) []byte
}

func benchmarkScenario(b *testing.B, sc requestScenario) {
	for _, wf := range sc.acceptContentTypes {
		b.Run(fmt.Sprintf("Format=%s", wf.name), func(b *testing.B) {
			b.ReportAllocs()

			client, server := setupAPIServer(b)
			ensureDefaultServiceAccount(b, client)
			pods := seedPods(b, client, sc.seedPods)

			reqContentType := sc.requestContentType
			if reqContentType == "" && sc.body != nil {
				reqContentType = wf.accept
			}
			fullPath := buildRequestPath(sc, pods)
			var payload []byte
			if sc.body != nil {
				var pod *v1.Pod
				if len(pods) > 0 {
					pod = pods[0]
				}
				payload = sc.body(wf.accept, pod)
			}

			var wireBytes int
			for b.Loop() {
				wireBytes = runHTTP(b, server, sc.httpMethod, fullPath, reqContentType, wf.accept, payload)
			}

			b.ReportMetric(float64(wireBytes), "wire-B/op")
		})
	}
}

func buildRequestPath(sc requestScenario, pods []*v1.Pod) string {
	if sc.query == "" {
		return sc.path
	}
	query := sc.query
	if len(pods) > 0 {
		initialRV := pods[0].ResourceVersion
		latestRV := pods[len(pods)-1].ResourceVersion
		query = strings.NewReplacer("FIRST_RV", initialRV, "LATEST_RV", latestRV).Replace(query)
	}
	return sc.path + "?" + query
}
