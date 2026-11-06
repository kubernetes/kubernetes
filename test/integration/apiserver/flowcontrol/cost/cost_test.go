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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
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
// post-GC retained heap/runtime memory for kube-apiserver operations:
//   - DryRun=true: concurrent read and dryRun=All write operations against N=1000 seeded pods.
//   - DryRun=false: sequential committed write operations fenced on etcd -> watchCache -> podInformer.
func BenchmarkRequestCost(b *testing.B) {
	framework.FixBenchmarkJSONOutput(b)
	featuregatetesting.SetFeatureGatesDuringTest(b, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CBORServingAndStorage:    true,
		features.DetectCacheInconsistency: false,
		features.WatchList:                true,
	})

	const (
		perfDashMetricName = "BenchmarkRequestCostV3"
		podCount           = 1000
		podPath            = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods/pod-000"
		unboundPodPath     = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods/pod-999"
		podsPath           = "/api/v1/namespaces/" + metav1.NamespaceDefault + "/pods"
		ephemeralPodPath   = "/api/v1/namespaces/" + ephemeralNamespace + "/pods/pod-000"
		ephemeralPodsPath  = "/api/v1/namespaces/" + ephemeralNamespace + "/pods"
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
			path:               ephemeralPodPath,
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
			path:               ephemeralPodsPath,
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
			query:              "watch=true&fieldSelector=metadata.name=pod-000",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv=""`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" labelSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&labelSelector=name=small-deployment-00",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" nodeSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&fieldSelector=spec.nodeName=kind-worker0",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 labelSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&labelSelector=name=small-deployment-00",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 nodeSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&fieldSelector=spec.nodeName=kind-worker0",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=LATEST_RV",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=LATEST_RV",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=FIRST_RV",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=FIRST_RV",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=FIRST_RV labelSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=FIRST_RV&labelSelector=name=small-deployment-00",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=FIRST_RV nodeSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=FIRST_RV&fieldSelector=spec.nodeName=kind-worker0",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" sendInitialEvents=false`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=false&resourceVersionMatch=NotOlderThan",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 sendInitialEvents=false",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=false&resourceVersionMatch=NotOlderThan",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" WatchList`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" WatchList labelSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&labelSelector=name=small-deployment-00",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            `rv="" WatchList nodeSelector`,
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&fieldSelector=spec.nodeName=kind-worker0",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 WatchList",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 WatchList labelSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&labelSelector=name=small-deployment-00",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "WATCH",
			options:            "rv=0 WatchList nodeSelector",
			httpMethod:         http.MethodGet,
			path:               podsPath,
			query:              "watch=true&resourceVersion=0&sendInitialEvents=true&resourceVersionMatch=NotOlderThan&allowWatchBookmarks=true&fieldSelector=spec.nodeName=kind-worker0",
			acceptContentTypes: watchFormats,
		},
		{
			method:             "POST",
			options:            "create",
			httpMethod:         http.MethodPost,
			path:               podsPath,
			query:              "fieldManager=bench",
			acceptContentTypes: writeFormats,
			body:               staticPodBody(b),
		},
		{
			method:             "POST",
			options:            "create admission=" + string(validatingPolicy),
			httpMethod:         http.MethodPost,
			path:               podsPath,
			query:              "fieldManager=bench",
			acceptContentTypes: writeFormats,
			body:               staticPodBody(b),
			admissionPolicy:    validatingPolicy,
		},
		{
			method:             "POST",
			options:            "create admission=" + string(mutatingApplyPolicy),
			httpMethod:         http.MethodPost,
			path:               podsPath,
			query:              "fieldManager=bench",
			acceptContentTypes: writeFormats,
			body:               staticPodBody(b),
			admissionPolicy:    mutatingApplyPolicy,
		},
		{
			method:             "POST",
			options:            "create admission=" + string(mutatingJSONPatchPolicy),
			httpMethod:         http.MethodPost,
			path:               podsPath,
			query:              "fieldManager=bench",
			acceptContentTypes: writeFormats,
			body:               staticPodBody(b),
			admissionPolicy:    mutatingJSONPatchPolicy,
		},
		{
			method:             "POST",
			options:            "create " + string(validatingPolicy),
			httpMethod:         http.MethodPost,
			path:               vapPath,
			query:              "fieldManager=bench",
			acceptContentTypes: writeFormats,
			body:               staticObjectBody(b, benchVAP()),
		},
		{
			method:             "POST",
			options:            "create " + string(mutatingApplyPolicy),
			httpMethod:         http.MethodPost,
			path:               mapPath,
			query:              "fieldManager=bench",
			acceptContentTypes: writeFormats,
			body:               staticObjectBody(b, benchMAPApply()),
		},
		{
			method:             "POST",
			options:            "create " + string(mutatingJSONPatchPolicy),
			httpMethod:         http.MethodPost,
			path:               mapPath,
			query:              "fieldManager=bench",
			acceptContentTypes: writeFormats,
			body:               staticObjectBody(b, benchMAPJSONPatch()),
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
			body: func(mediaType string, pod *v1.Pod, alt int) []byte {
				updated := pod.DeepCopy()
				updated.UID = ""
				updated.ResourceVersion = ""
				updated.ManagedFields = nil
				if updated.Annotations == nil {
					updated.Annotations = map[string]string{}
				}
				updated.Annotations["bench.k8s.io/patch"] = strconv.Itoa(alt)
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
			body: func(mediaType string, pod *v1.Pod, alt int) []byte {
				updated := pod.DeepCopy()
				updated.UID = ""
				updated.ResourceVersion = ""
				updated.ManagedFields = nil
				updated.Status.Message = fmt.Sprintf("updated-status-%d", alt)
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
			body: staticBytesBody(
				`{"metadata":{"annotations":{"bench.k8s.io/patch":"0"}}}`,
				`{"metadata":{"annotations":{"bench.k8s.io/patch":"1"}}}`,
			),
		},
		{
			method:             "PATCH",
			options:            "spec tolerations",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench",
			requestContentType: string(types.StrategicMergePatchType),
			acceptContentTypes: writeFormats,
			body: staticBytesBody(
				`{"spec":{"tolerations":[{"key":"node.kubernetes.io/not-ready","operator":"Exists","effect":"NoExecute","tolerationSeconds":900},{"key":"node.kubernetes.io/unreachable","operator":"Exists","effect":"NoExecute","tolerationSeconds":900},{"key":"bench.k8s.io/taint","operator":"Exists","effect":"NoSchedule"}]}}`,
				`{"spec":{"tolerations":[{"key":"node.kubernetes.io/not-ready","operator":"Exists","effect":"NoExecute","tolerationSeconds":901},{"key":"node.kubernetes.io/unreachable","operator":"Exists","effect":"NoExecute","tolerationSeconds":900},{"key":"bench.k8s.io/taint","operator":"Exists","effect":"NoSchedule"}]}}`,
			),
		},
		{
			method:             "PATCH",
			options:            "merge-patch",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench",
			requestContentType: string(types.MergePatchType),
			acceptContentTypes: writeFormats,
			body: staticBytesBody(
				`{"metadata":{"annotations":{"bench.k8s.io/patch":"0"}}}`,
				`{"metadata":{"annotations":{"bench.k8s.io/patch":"1"}}}`,
			),
		},
		{
			method:             "PATCH",
			options:            "json-patch",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench",
			requestContentType: string(types.JSONPatchType),
			acceptContentTypes: writeFormats,
			body: staticBytesBody(
				`[{"op":"add","path":"/metadata/annotations/bench.k8s.io~1patch","value":"0"}]`,
				`[{"op":"add","path":"/metadata/annotations/bench.k8s.io~1patch","value":"1"}]`,
			),
		},
		{
			method:             "PATCH",
			options:            "apply",
			httpMethod:         http.MethodPatch,
			path:               podPath,
			query:              "fieldManager=bench&force=true",
			requestContentType: string(types.ApplyYAMLPatchType),
			acceptContentTypes: writeFormats,
			body: staticBytesBody(
				`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"pod-000","annotations":{"bench.k8s.io/patch":"0"}}}`,
				`{"apiVersion":"v1","kind":"Pod","metadata":{"name":"pod-000","annotations":{"bench.k8s.io/patch":"1"}}}`,
			),
		},
		{
			method:             "PATCH",
			options:            "status strategic-merge",
			httpMethod:         http.MethodPatch,
			path:               podPath + "/status",
			query:              "fieldManager=kubelet",
			requestContentType: string(types.StrategicMergePatchType),
			acceptContentTypes: writeFormats,
			body: staticBytesBody(
				`{"status":{"message":"updated-status-0"}}`,
				`{"status":{"message":"updated-status-1"}}`,
			),
		},
		{
			method:             "PATCH",
			options:            "status kubelet",
			httpMethod:         http.MethodPatch,
			path:               podPath + "/status",
			query:              "fieldManager=kubelet",
			requestContentType: string(types.StrategicMergePatchType),
			acceptContentTypes: writeFormats,
			body: staticBytesBody(
				`{"status":{"phase":"Running","podIP":"10.244.7.83","podIPs":[{"ip":"10.244.7.83"}],"conditions":[{"type":"Ready","status":"True","reason":"ContainersReady","lastTransitionTime":"2026-04-23T12:24:00Z"}],"containerStatuses":[{"name":"main","ready":true,"started":true,"restartCount":0,"image":"registry.k8s.io/pause:3.9","imageID":"registry.k8s.io/pause@sha256:1234","containerID":"containerd://main0","state":{"running":{"startedAt":"2026-04-23T12:24:00Z"}}}]}}`,
				`{"status":{"phase":"Running","podIP":"10.244.7.83","podIPs":[{"ip":"10.244.7.83"}],"conditions":[{"type":"Ready","status":"True","reason":"ContainersReady","lastTransitionTime":"2026-04-23T12:24:00Z"}],"containerStatuses":[{"name":"main","ready":true,"started":true,"restartCount":0,"image":"registry.k8s.io/pause:3.9","imageID":"registry.k8s.io/pause@sha256:1234","containerID":"containerd://main1","state":{"running":{"startedAt":"2026-04-23T12:24:00Z"}}}]}}`,
			),
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
		tCtx, server, informer := setupAPIServer(tCtx)
		runtime.ReadMemStats(&afterSetupMem)
		setupAllocBytes := float64(afterSetupMem.TotalAlloc - beforeSetupMem.TotalAlloc)

		ensureNamespace(tCtx, metav1.NamespaceDefault)
		ensureNamespace(tCtx, ephemeralNamespace)
		pods := seedPods(tCtx, informer, metav1.NamespaceDefault, podCount, 0)

		// Run scenarios that seed 45KB pods (extraBytes > 0) last so their
		// watchCache history does not inflate retained heap or advance RV ahead
		// of LATEST_RV.
		for _, largePayload := range []bool{false, true} {
			for _, sc := range scenarios {
				if (sc.extraBytes > 0) != largePayload {
					continue
				}
				b.Run(fmt.Sprintf("Method=%s/Options=%s", sc.method, sc.options), func(b *testing.B) {
					framework.FixBenchmarkJSONOutput(b)
					targetPods := pods
					if sc.extraBytes > 0 {
						count := podCount
						if sc.method == "GET" {
							count = 1
						}
						targetPods = seedPods(tCtx, informer, ephemeralNamespace, count, sc.extraBytes)
						defer clearNamespace(tCtx, informer, ephemeralNamespace)
					}
					if sc.admissionPolicy != admissionPolicyNone {
						defer installAdmissionPolicy(tCtx, server, sc.admissionPolicy)()
					}
					for _, dryRun := range []bool{true, false} {
						if !dryRun && !supportsCommit(sc) {
							continue
						}
						b.Run(fmt.Sprintf("DryRun=%t", dryRun), func(b *testing.B) {
							framework.FixBenchmarkJSONOutput(b)
							report.DataItems = append(report.DataItems, benchmarkScenario(b, tCtx, server, informer, sc, targetPods, setupAllocBytes, dryRun)...)
						})
					}
				})
			}
		}
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
	body               bodyBuilder
	// admissionPolicy, if set, is installed while the scenario is measured.
	admissionPolicy admissionPolicy
}

func supportsCommit(sc requestScenario) bool {
	return sc.httpMethod == http.MethodPut ||
		sc.httpMethod == http.MethodPatch ||
		(sc.httpMethod == http.MethodPost && (sc.options == "create" || sc.admissionPolicy != admissionPolicyNone))
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

func benchmarkScenario(b *testing.B, tCtx ktesting.TContext, server http.Handler, informer *informerWaiter, sc requestScenario, pods []*v1.Pod, setupAllocBytes float64, dryRun bool) []dataItem {
	var items []dataItem
	dryRunLabel := "DryRun=False"
	if dryRun {
		dryRunLabel = "DryRun=True"
	}
	optionsLabel := fmt.Sprintf("%s %s", sc.method, sc.options)

	for _, wf := range sc.acceptContentTypes {
		// b.Run invokes its callback multiple times while calibrating b.N;
		// overwrite formatItems on each pass so only the final calibrated
		// run is appended to the report.
		var formatItems []dataItem
		b.Run(fmt.Sprintf("Format=%s", wf.name), func(b *testing.B) {
			framework.FixBenchmarkJSONOutput(b)
			b.ReportAllocs()

			var (
				reqBytesPerOp      float64
				totalResponseBytes int64
				startMem, endMem   runtime.MemStats
			)
			if dryRun {
				reqBytesPerOp, totalResponseBytes = runScenarioDryRun(b, server, sc, wf, pods, &startMem, &endMem)
			} else {
				reqBytesPerOp, totalResponseBytes = runScenarioCommitted(b, tCtx, server, informer, sc, wf, &startMem, &endMem)
			}

			var postGCMem runtime.MemStats
			// Flush both generations of Go's sync.Pool victim cache before sampling retained heap.
			runtime.GC()
			debug.FreeOSMemory()
			runtime.ReadMemStats(&postGCMem)

			respBytesPerOp := float64(totalResponseBytes) / float64(b.N)
			metrics := []struct {
				benchUnit string
				name      string
				unit      string
				val       float64
			}{
				{"", "Allocated bytes per operation", "B", float64(endMem.TotalAlloc-startMem.TotalAlloc) / float64(b.N)},
				{"", "Allocations per operation", "1", float64(endMem.Mallocs-startMem.Mallocs) / float64(b.N)},
				{"", "Duration per operation", "ns", float64(b.Elapsed().Nanoseconds()) / float64(b.N)},
				{"request-B/op", "Request body bytes per operation", "B", reqBytesPerOp},
				{"response-B/op", "Response body bytes per operation", "B", respBytesPerOp},
				{"setup-alloc-B", "Server setup allocated bytes", "B", setupAllocBytes},
				{"heap-inuse-B", "Retained heap inuse bytes", "B", float64(postGCMem.HeapInuse)},
				{"heap-alloc-B", "Retained heap alloc bytes", "B", float64(postGCMem.HeapAlloc)},
				{"runtime-inuse-B", "Retained runtime inuse bytes", "B", float64(postGCMem.Sys - postGCMem.HeapReleased)},
				{"heap-released-B", "Heap released bytes", "B", float64(postGCMem.HeapReleased)},
				{"heap-frag-B", "Heap span fragmentation bytes", "B", float64(postGCMem.HeapInuse - postGCMem.HeapAlloc)},
			}

			formatItems = make([]dataItem, 0, len(metrics))
			for _, m := range metrics {
				if m.benchUnit != "" {
					b.ReportMetric(m.val, m.benchUnit)
				}
				formatItems = append(formatItems, dataItem{
					Labels: map[string]string{
						"DryRun":  dryRunLabel,
						"Options": optionsLabel,
						"Format":  wf.name,
						"Metric":  m.name,
					},
					Unit: m.unit,
					Data: map[string]float64{wf.name: m.val},
				})
			}
		})
		items = append(items, formatItems...)
	}

	return items
}

func runScenarioDryRun(b *testing.B, server http.Handler, sc requestScenario, wf contentType, pods []*v1.Pod, startMem, endMem *runtime.MemStats) (float64, int64) {
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
		payload = sc.body(wf.accept, pods[0], 0)
	}
	// Warm lazy schema, converter, sync.Pool, and per-namespace admission
	// caches before measurements start.
	warmupPath := fullPath
	if sc.method == "WATCH" {
		warmupPath += "&timeoutSeconds=1"
	}
	sink := newResponseSink(0, nil)
	runHTTP(context.Background(), b, server, sc.httpMethod, warmupPath, reqContentType, wf.accept, payload, sink)
	// Instead of waiting for request timeout let's close after same number of writes. Number of writes is more reliable than bytes written due to non-deterministic response size of Format=table.
	expectedNumberOfWrites := sink.writes

	var totalResponseBytes atomic.Int64
	runtime.ReadMemStats(startMem)
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var cancel context.CancelFunc = func() {}
		var ctx context.Context = context.Background()
		for pb.Next() {
			var sink *responseSink
			if sc.method == "WATCH" {
				ctx, cancel = context.WithCancel(context.Background())
				sink = newResponseSink(expectedNumberOfWrites, cancel)
			} else {
				sink = newResponseSink(0, nil)
			}
			n := runHTTP(ctx, b, server, sc.httpMethod, fullPath, reqContentType, wf.accept, payload, sink)
			cancel()
			totalResponseBytes.Add(int64(n))
		}
	})
	b.StopTimer()
	runtime.ReadMemStats(endMem)
	return float64(len(payload)), totalResponseBytes.Load()
}

func runScenarioCommitted(b *testing.B, tCtx ktesting.TContext, server http.Handler, informer *informerWaiter, sc requestScenario, wf contentType, startMem, endMem *runtime.MemStats) (float64, int64) {
	defer clearNamespace(tCtx, informer, ephemeralNamespace)

	reqContentType := sc.requestContentType
	if reqContentType == "" && sc.body != nil {
		reqContentType = wf.accept
	}
	var targetPod *v1.Pod
	if sc.httpMethod != http.MethodPost {
		targetPod = seedPods(tCtx, informer, ephemeralNamespace, 1, 0)[0]
	}
	sc.path = strings.Replace(sc.path, "/namespaces/"+metav1.NamespaceDefault+"/", "/namespaces/"+ephemeralNamespace+"/", 1)
	fullPath := buildRequestPath(sc, nil)

	// Alternate between two payloads so etcd3.store.GuaranteedUpdate never
	// short-circuits on byte-identical serialized state.
	var payloads [2][]byte
	if sc.body != nil {
		payloads[0] = sc.body(wf.accept, targetPod, 0)
		payloads[1] = sc.body(wf.accept, targetPod, 1)
	}

	ctx := b.Context()
	sink := newResponseSink(0, nil)
	runHTTP(ctx, b, server, sc.httpMethod, fullPath, reqContentType, wf.accept, payloads[1], sink)
	informer.wait(ctx, b, 1)
	if sc.httpMethod == http.MethodPost {
		clearNamespace(tCtx, informer, ephemeralNamespace)
	} else {
		informer.resetEvents()
	}

	var totalRequestBytes, totalResponseBytes int64
	runtime.ReadMemStats(startMem)
	b.ResetTimer()
	for i := range b.N {
		payload := payloads[i%2]
		totalRequestBytes += int64(len(payload))
		sink := newResponseSink(0, nil)
		totalResponseBytes += int64(runHTTP(ctx, b, server, sc.httpMethod, fullPath, reqContentType, wf.accept, payload, sink))
	}
	informer.wait(ctx, b, int64(b.N))
	b.StopTimer()
	runtime.ReadMemStats(endMem)
	return float64(totalRequestBytes) / float64(b.N), totalResponseBytes
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
