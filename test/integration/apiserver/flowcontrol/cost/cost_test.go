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
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/ktesting"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// BenchmarkRequestCost measures per-request heap allocations (B/op and allocs/op),
// request duration, HTTP request/response sizes, server setup allocations, and
// post-GC retained heap/runtime memory for concurrent kube-apiserver read and
// dry-run write operations against N=1000 seeded exemplar pods.
func BenchmarkRequestCost(b *testing.B) {
	framework.FixBenchmarkJSONOutput(b)
	featuregatetesting.SetFeatureGatesDuringTest(b, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CBORServingAndStorage:    true,
		features.DetectCacheInconsistency: false,
		features.WatchList:                true,
	})

	const (
		perfDashMetricName = "BenchmarkRequestCostV2"
		podCount           = 1000
		podPath            = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods/pod-000"
		unboundPodPath     = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods/pod-999"
		podsPath           = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods"
		largePodPath       = "/api/v1/namespaces/" + largePodNamespace + "/pods/pod-000"
		largePodsPath      = "/api/v1/namespaces/" + largePodNamespace + "/pods"
	)

	scenarios := []requestScenario{
		{
			method:             "GET",
			options:            `rv=""`,
			httpMethod:         http.MethodGet,
			path:               podPath,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "GET",
			options:            "rv=0",
			httpMethod:         http.MethodGet,
			path:               podPath,
			query:              "resourceVersion=0",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "GET",
			options:            "rv=0 45KB",
			httpMethod:         http.MethodGet,
			path:               largePodPath,
			query:              "resourceVersion=0",
			extraBytes:         35 * 1024,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            `rv=""`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            `rv="" labelSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "labelSelector=name=small-deployment-00",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            `rv="" nodeSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "fieldSelector=spec.nodeName=kind-worker0",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "rv=0",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=0",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "rv=0 45KB",
			httpMethod:         http.MethodGet,
			path:               largePodsPath,
			query:              "resourceVersion=0",
			extraBytes:         35 * 1024,
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "rv=0 labelSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=0&labelSelector=name=small-deployment-00",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "rv=0 nodeSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=0&fieldSelector=spec.nodeName=kind-worker0",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "rv=Exact",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=LATEST_RV&resourceVersionMatch=Exact",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "rv=Exact labelSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=LATEST_RV&resourceVersionMatch=Exact&labelSelector=name=small-deployment-00",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "rv=Exact nodeSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=LATEST_RV&resourceVersionMatch=Exact&fieldSelector=spec.nodeName=kind-worker0",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "LIST",
			options:            "limit=100",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "limit=100",
			acceptContentTypes: allReadFormats,
		},
		{
			method:             "WATCH",
			options:            "fieldSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&fieldSelector=metadata.name=pod-000&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv=""`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" labelSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&labelSelector=name=small-deployment-00&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" nodeSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&fieldSelector=spec.nodeName=kind-worker0&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 labelSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=0&labelSelector=name=small-deployment-00&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 nodeSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "resourceVersion=0&fieldSelector=spec.nodeName=kind-worker0&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=LATEST_RV",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=LATEST_RV&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=FIRST_RV",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=FIRST_RV&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=FIRST_RV labelSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=FIRST_RV&labelSelector=name=small-deployment-00&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=FIRST_RV nodeSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=FIRST_RV&fieldSelector=spec.nodeName=kind-worker0&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" sendInitialEvents=false`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=false&resourceVersionMatch=NotOlderThan&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 sendInitialEvents=false",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=false&resourceVersionMatch=NotOlderThan&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" WatchList`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" WatchList labelSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&labelSelector=name=small-deployment-00&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" WatchList nodeSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&fieldSelector=spec.nodeName=kind-worker0&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 WatchList",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 WatchList labelSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&labelSelector=name=small-deployment-00&timeoutSeconds=1",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 WatchList nodeSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&fieldSelector=spec.nodeName=kind-worker0&timeoutSeconds=1",
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
			options:            "binding",
			httpMethod:         http.MethodPost,
			path:               unboundPodPath + "/binding",
			query:              "fieldManager=bench",
			acceptContentTypes: writeFormats,
			body: staticObjectBody(b, &v1.Binding{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:   metav1.NamespaceDefault,
					Name:        "pod-999",
					Annotations: map[string]string{"bench.k8s.io/patch": "true"},
				},
				Target: v1.ObjectReference{
					Kind: "Node",
					Name: "kind-worker6",
				},
			}),
		},
		{
			method:             "POST",
			options:            "eviction",
			httpMethod:         http.MethodPost,
			path:               podPath + "/eviction",
			acceptContentTypes: writeFormats,
			body: staticObjectBody(b, &policyv1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "pod-000",
				},
			}),
		},
		{
			method:             "PUT",
			options:            "update",
			httpMethod:         http.MethodPut,
			path:               podPath,
			query:              "fieldManager=bench",
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
			requestContentType: string(types.StrategicMergePatchType),
			acceptContentTypes: writeFormats,
			body:               staticBytesBody(`{"metadata":{"annotations":{"bench.k8s.io/patch":"true"}}}`),
		},
		{
			method:             "PATCH",
			options:            "spec tolerations",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench",
			requestContentType: string(types.StrategicMergePatchType),
			acceptContentTypes: writeFormats,
			body:               staticBytesBody(`{"spec":{"tolerations":[{"key":"node.kubernetes.io/not-ready","operator":"Exists","effect":"NoExecute","tolerationSeconds":900},{"key":"node.kubernetes.io/unreachable","operator":"Exists","effect":"NoExecute","tolerationSeconds":900},{"key":"bench.k8s.io/taint","operator":"Exists","effect":"NoSchedule"}]}}`),
		},
		{
			method:             "PATCH",
			options:            "merge-patch",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench",
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
			requestContentType: string(types.StrategicMergePatchType),
			acceptContentTypes: writeFormats,
			body:               staticBytesBody(`{"status":{"message":"updated-status"}}`),
		},
		{
			method:             "PATCH",
			options:            "status kubelet",
			httpMethod:         http.MethodPatch,
			path:               podPath + "/status",
			query:              "fieldManager=kubelet",
			requestContentType: string(types.StrategicMergePatchType),
			acceptContentTypes: writeFormats,
			body:               staticBytesBody(`{"status":{"phase":"Running","podIP":"10.244.7.83","podIPs":[{"ip":"10.244.7.83"}],"conditions":[{"type":"Ready","status":"True","reason":"ContainersReady","lastTransitionTime":"2026-04-23T12:24:00Z"}],"containerStatuses":[{"name":"main","ready":true,"started":true,"restartCount":0,"image":"registry.k8s.io/pause:3.9","imageID":"registry.k8s.io/pause@sha256:1234","containerID":"containerd://main1","state":{"running":{"startedAt":"2026-04-23T12:24:00Z"}}}]}}`),
		},
		{
			method:             "DELETE",
			options:            "gracePeriod=0",
			httpMethod:         http.MethodDelete,
			path:               podPath,
			query:              "gracePeriodSeconds=0",
			acceptContentTypes: writeFormats,
		},
		{
			method:             "DELETECOLLECTION",
			options:            "gracePeriod=0",
			httpMethod:         http.MethodDelete,
			path:               podsPath,
			query:              "gracePeriodSeconds=0",
			acceptContentTypes: writeFormats,
		},
	}

	report := perfData{Version: "v1"}
	b.Run(fmt.Sprintf("N=%d", podCount), func(b *testing.B) {
		framework.FixBenchmarkJSONOutput(b)
		tCtx := ktesting.Init(b)

		var beforeSetupMem, afterSetupMem runtime.MemStats
		runtime.ReadMemStats(&beforeSetupMem)
		tCtx, server, podInformer := setupAPIServer(tCtx)
		runtime.ReadMemStats(&afterSetupMem)
		setupAllocBytes := float64(afterSetupMem.TotalAlloc - beforeSetupMem.TotalAlloc)

		ensureNamespace(tCtx, metav1.NamespaceDefault)
		pods := seedPods(tCtx, metav1.NamespaceDefault, podCount, 0)
		waitForResourceVersion(tCtx, podInformer, pods[len(pods)-1].ResourceVersion)

		b.Run("DryRun=true", func(b *testing.B) {
			framework.FixBenchmarkJSONOutput(b)
			// Seed 45KB pods in a separate namespace after default-namespace scenarios
			// so their watchCache state does not inflate retained heap or advance RV
			// ahead of LATEST_RV.
			seedLargePods := sync.OnceValue(func() []*v1.Pod {
				ensureNamespace(tCtx, largePodNamespace)
				p := seedPods(tCtx, largePodNamespace, podCount, 35*1024)
				waitForResourceVersion(tCtx, podInformer, p[len(p)-1].ResourceVersion)
				return p
			})
			for _, largePayload := range []bool{false, true} {
				for _, sc := range scenarios {
					if (sc.extraBytes > 0) != largePayload {
						continue
					}
					b.Run(fmt.Sprintf("Method=%s/Options=%s", sc.method, sc.options), func(b *testing.B) {
						framework.FixBenchmarkJSONOutput(b)
						targetPods := pods
						if largePayload {
							targetPods = seedLargePods()
						}
						report.DataItems = append(report.DataItems, benchmarkScenario(b, server, sc, targetPods, podCount, setupAllocBytes)...)
					})
				}
			}
		})
	})
	if artifacts := os.Getenv("ARTIFACTS"); artifacts != "" && len(report.DataItems) > 0 {
		writePerfData(b, artifacts, perfDashMetricName, report)
	}
}

type requestScenario struct {
	method             string
	options            string
	httpMethod         string
	path               string
	query              string
	extraBytes         int
	requestContentType string
	acceptContentTypes []contentType
	body               func(mediaType string, pod *v1.Pod) []byte
}

// perfData and dataItem match k8s.io/kubernetes/test/e2e/perftype (duplicated
// because test/integration/.import-restrictions forbids importing test/e2e).
type perfData struct {
	Version   string     `json:"version"`
	DataItems []dataItem `json:"dataItems"`
}

type dataItem struct {
	Data   map[string]float64 `json:"data"`
	Unit   string             `json:"unit"`
	Labels map[string]string  `json:"labels,omitempty"`
}

func benchmarkScenario(b *testing.B, server http.Handler, sc requestScenario, pods []*v1.Pod, podCount int, setupAllocBytes float64) []dataItem {
	var items []dataItem
	nStr := strconv.Itoa(podCount)

	for _, wf := range sc.acceptContentTypes {
		b.Run(fmt.Sprintf("Format=%s", wf.name), func(b *testing.B) {
			framework.FixBenchmarkJSONOutput(b)
			b.ReportAllocs()

			reqContentType := sc.requestContentType
			if reqContentType == "" && sc.body != nil {
				reqContentType = wf.accept
			}
			fullPath := buildRequestPath(sc, pods)
			if sc.httpMethod != http.MethodGet {
				fullPath = withDryRun(fullPath)
			}
			var payload []byte
			if sc.body != nil {
				payload = sc.body(wf.accept, pods[0])
			}

			// Warm lazy schema, converter, sync.Pool, and per-namespace admission
			// caches before measurements start.
			runHTTP(b, server, sc.httpMethod, fullPath, reqContentType, wf.accept, payload)

			var (
				startMem, endMem, postGCMem runtime.MemStats
				totalResponseBytes          atomic.Int64
			)
			runtime.ReadMemStats(&startMem)
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					n := runHTTP(b, server, sc.httpMethod, fullPath, reqContentType, wf.accept, payload)
					totalResponseBytes.Add(int64(n))
				}
			})
			b.StopTimer()
			runtime.ReadMemStats(&endMem)
			debug.FreeOSMemory()
			runtime.ReadMemStats(&postGCMem)

			respBytesPerOp := float64(totalResponseBytes.Load()) / float64(b.N)
			metrics := []struct {
				benchUnit string
				name      string
				unit      string
				val       float64
			}{
				{"", "Allocated bytes per operation", "B", float64(endMem.TotalAlloc-startMem.TotalAlloc) / float64(b.N)},
				{"", "Allocations per operation", "1", float64(endMem.Mallocs-startMem.Mallocs) / float64(b.N)},
				{"", "Duration per operation", "ns", float64(b.Elapsed().Nanoseconds()) / float64(b.N)},
				{"request-B/op", "Request body bytes per operation", "B", float64(len(payload))},
				{"response-B/op", "Response body bytes per operation", "B", respBytesPerOp},
				{"setup-alloc-B", "Server setup allocated bytes", "B", setupAllocBytes},
				{"heap-inuse-B", "Retained heap inuse bytes", "B", float64(postGCMem.HeapInuse)},
				{"heap-alloc-B", "Retained heap alloc bytes", "B", float64(postGCMem.HeapAlloc)},
				{"runtime-inuse-B", "Retained runtime inuse bytes", "B", float64(postGCMem.Sys - postGCMem.HeapReleased)},
				{"heap-released-B", "Heap released bytes", "B", float64(postGCMem.HeapReleased)},
				{"heap-frag-B", "Heap span fragmentation bytes", "B", float64(postGCMem.HeapInuse - postGCMem.HeapAlloc)},
			}

			for _, m := range metrics {
				if m.benchUnit != "" {
					b.ReportMetric(m.val, m.benchUnit)
				}
				items = append(items, dataItem{
					Labels: map[string]string{
						"N":       nStr,
						"DryRun":  "true",
						"Method":  sc.method,
						"Options": sc.options,
						"Format":  wf.name,
						"Metric":  m.name,
					},
					Unit: m.unit,
					Data: map[string]float64{wf.name: m.val},
				})
			}
		})
	}

	return items
}

func writePerfData(tb testing.TB, artifactsDir, metricName string, data perfData) {
	tb.Helper()
	if err := os.MkdirAll(artifactsDir, 0750); err != nil {
		tb.Fatalf("failed to create artifacts dir %s: %v", artifactsDir, err)
	}
	path := filepath.Join(artifactsDir, fmt.Sprintf("GenericPrometheusQuery %s_%s.json", metricName, time.Now().UTC().Format("2006-01-02T15:04:05Z")))
	raw, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		tb.Fatalf("failed to marshal perf data: %v", err)
	}
	if err := os.WriteFile(path, raw, 0644); err != nil {
		tb.Fatalf("failed to write perf data to %s: %v", path, err)
	}
	tb.Logf("PerfData report written to: %s", path)
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

func withDryRun(path string) string {
	if strings.Contains(path, "?") {
		return path + "&dryRun=All"
	}
	return path + "?dryRun=All"
}
