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

package storage

import (
	"bytes"
	"context"
	_ "embed"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/client-go/applyconfigurations"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	podstore "k8s.io/kubernetes/pkg/registry/core/pod/storage"
	"k8s.io/kubernetes/test/integration/framework"
	"sigs.k8s.io/structured-merge-diff/v7/fieldpath"
)

//go:embed testdata/exemplar_pod.yaml
var exemplarPodYAML []byte

type podBenchmark struct {
	rest        *podstore.REST
	statusREST  *podstore.StatusREST
	scope       *handlers.RequestScope
	statusScope *handlers.RequestScope
	admit       admission.Interface
	codec       runtime.Codec
	pod         *api.Pod
}

type podRequest struct {
	ctx    context.Context
	target string
	// Rebuilt per request: the endpoint handlers reassign their captured admit
	// parameter, so reusing one handler re-wraps the admission chain every call.
	handler func() http.HandlerFunc
}

func BenchmarkPatchPod(b *testing.B) {
	bench := setupPodBenchmark(b)
	pod := bench.pod
	req := podRequest{
		ctx:    podRequestContext(pod, "patch", ""),
		target: fmt.Sprintf("/api/v1/namespaces/%s/pods/%s?fieldManager=patch-manager", pod.Namespace, pod.Name),
		handler: func() http.HandlerFunc {
			return handlers.PatchResource(bench.rest, bench.scope, bench.admit, []string{string(types.StrategicMergePatchType)})
		},
	}
	doRequest(b, req, http.MethodPatch, string(types.StrategicMergePatchType), labelPatch(0), http.StatusOK)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doRequest(b, req, http.MethodPatch, string(types.StrategicMergePatchType), labelPatch(i+1), http.StatusOK)
	}
}

func BenchmarkPatchPodStatus(b *testing.B) {
	bench := setupPodBenchmark(b)
	pod := bench.pod
	req := podRequest{
		ctx:    podRequestContext(pod, "patch", "status"),
		target: fmt.Sprintf("/api/v1/namespaces/%s/pods/%s/status?fieldManager=kubelet", pod.Namespace, pod.Name),
		handler: func() http.HandlerFunc {
			return handlers.PatchResource(bench.statusREST, bench.statusScope, bench.admit, []string{string(types.StrategicMergePatchType)})
		},
	}
	doRequest(b, req, http.MethodPatch, string(types.StrategicMergePatchType), conditionsPatch(0), http.StatusOK)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doRequest(b, req, http.MethodPatch, string(types.StrategicMergePatchType), conditionsPatch(i+1), http.StatusOK)
	}
}

func BenchmarkUpdatePod(b *testing.B) {
	bench := setupPodBenchmark(b)
	pod := bench.pod
	req := podRequest{
		ctx:     podRequestContext(pod, "update", ""),
		target:  fmt.Sprintf("/api/v1/namespaces/%s/pods/%s?fieldManager=patch-manager", pod.Namespace, pod.Name),
		handler: func() http.HandlerFunc { return handlers.UpdateResource(bench.rest, bench.scope, bench.admit) },
	}

	updated := pod.DeepCopy()
	if updated.Labels == nil {
		updated.Labels = map[string]string{}
	}
	advance := func(w *httptest.ResponseRecorder, i int) []byte {
		updated.ResourceVersion = decodePod(b, bench.codec, w).ResourceVersion
		updated.Labels["bench-updated"] = strconv.Itoa(i)
		return encodePod(b, bench.codec, updated)
	}

	body := advance(doRequest(b, req, http.MethodPut, runtime.ContentTypeProtobuf, encodePod(b, bench.codec, updated), http.StatusOK), 1)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := doRequest(b, req, http.MethodPut, runtime.ContentTypeProtobuf, body, http.StatusOK)
		b.StopTimer()
		body = advance(w, i+2)
		b.StartTimer()
	}
}

func BenchmarkCreatePod(b *testing.B) {
	bench := setupPodBenchmark(b)
	pod := bench.pod
	req := podRequest{
		ctx:     podRequestContext(pod, "create", ""),
		target:  fmt.Sprintf("/api/v1/namespaces/%s/pods?fieldManager=bench-client", pod.Namespace),
		handler: func() http.HandlerFunc { return handlers.CreateResource(bench.rest, bench.scope, bench.admit) },
	}
	body := podCreateBody(b, bench.codec)
	deletePod(b, bench)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doRequest(b, req, http.MethodPost, runtime.ContentTypeProtobuf, body, http.StatusCreated)
		b.StopTimer()
		deletePod(b, bench)
		b.StartTimer()
	}
}

func BenchmarkDeletePod(b *testing.B) {
	bench := setupPodBenchmark(b)
	pod := bench.pod
	// gracePeriodSeconds=0 keeps this a real delete rather than a graceful
	// deletion, which would only stamp deletionTimestamp and leave the key.
	req := podRequest{
		ctx:     podRequestContext(pod, "delete", ""),
		target:  fmt.Sprintf("/api/v1/namespaces/%s/pods/%s?gracePeriodSeconds=0", pod.Namespace, pod.Name),
		handler: func() http.HandlerFunc { return handlers.DeleteResource(bench.rest, true, bench.scope, bench.admit) },
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		doRequest(b, req, http.MethodDelete, "", nil, http.StatusOK)
		b.StopTimer()
		seedPod(b, bench)
		b.StartTimer()
	}
}

func doRequest(b *testing.B, r podRequest, method, contentType string, body []byte, wantStatus int) *httptest.ResponseRecorder {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req := httptest.NewRequestWithContext(r.ctx, method, r.target, reader)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	req.Header.Set("Accept", runtime.ContentTypeProtobuf)
	w := httptest.NewRecorder()
	r.handler()(w, req)
	if w.Code != wantStatus {
		b.Fatalf("unexpected status %d for %s %s: %s", w.Code, method, r.target, w.Body.String())
	}
	return w
}

func labelPatch(i int) []byte {
	return fmt.Appendf(nil, `{"metadata":{"labels":{"bench-updated":"%d"}}}`, i)
}

func conditionsPatch(i int) []byte {
	readyStatus := "True"
	if i%2 == 0 {
		readyStatus = "False"
	}
	return fmt.Appendf(nil,
		`{"status":{"conditions":[{"type":"Ready","status":%q,"message":"kubelet-probe-%d"},{"type":"ContainersReady","status":%q}]}}`,
		readyStatus, i, readyStatus)
}

func podRequestContext(pod *api.Pod, verb, subresource string) context.Context {
	return audit.WithAuditContext(request.WithNamespace(request.WithRequestInfo(context.Background(), &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              verb,
		APIVersion:        "v1",
		Resource:          "pods",
		Subresource:       subresource,
		Namespace:         pod.Namespace,
		Name:              pod.Name,
	}), pod.Namespace))
}

func podFixture(b *testing.B) *api.Pod {
	b.Helper()
	var v1Pod v1.Pod
	if err := yaml.Unmarshal(exemplarPodYAML, &v1Pod); err != nil {
		b.Fatalf("failed to unmarshal exemplar_pod.yaml: %v", err)
	}
	var internalPod api.Pod
	if err := legacyscheme.Scheme.Convert(&v1Pod, &internalPod, nil); err != nil {
		b.Fatalf("failed to convert pod: %v", err)
	}
	return &internalPod
}

func podCreateBody(b *testing.B, codec runtime.Codec) []byte {
	b.Helper()
	pod := podFixture(b)
	pod.Status = api.PodStatus{}
	pod.ManagedFields = nil
	pod.UID = ""
	pod.ResourceVersion = ""
	pod.CreationTimestamp = metav1.Time{}
	return encodePod(b, codec, pod)
}

func encodePod(b *testing.B, codec runtime.Codec, pod *api.Pod) []byte {
	b.Helper()
	data, err := runtime.Encode(codec, pod)
	if err != nil {
		b.Fatalf("failed to encode pod: %v", err)
	}
	return data
}

func decodePod(b *testing.B, codec runtime.Codec, w *httptest.ResponseRecorder) *api.Pod {
	b.Helper()
	var pod api.Pod
	if err := runtime.DecodeInto(codec, w.Body.Bytes(), &pod); err != nil {
		b.Fatalf("failed to decode pod from response: %v", err)
	}
	return &pod
}

// Status needs a second request because podStrategy.PrepareForCreate resets it.
func seedPod(b *testing.B, bench *podBenchmark) *api.Pod {
	b.Helper()
	fixture := podFixture(b)
	created := decodePod(b, bench.codec, doRequest(b, podRequest{
		ctx:     podRequestContext(fixture, "create", ""),
		target:  fmt.Sprintf("/api/v1/namespaces/%s/pods?fieldManager=bench-client", fixture.Namespace),
		handler: func() http.HandlerFunc { return handlers.CreateResource(bench.rest, bench.scope, bench.admit) },
	}, http.MethodPost, runtime.ContentTypeProtobuf, podCreateBody(b, bench.codec), http.StatusCreated))

	created.Status = fixture.Status
	return decodePod(b, bench.codec, doRequest(b, podRequest{
		ctx:    podRequestContext(created, "update", "status"),
		target: fmt.Sprintf("/api/v1/namespaces/%s/pods/%s/status?fieldManager=kubelet", created.Namespace, created.Name),
		handler: func() http.HandlerFunc {
			return handlers.UpdateResource(bench.statusREST, bench.statusScope, bench.admit)
		},
	}, http.MethodPut, runtime.ContentTypeProtobuf, encodePod(b, bench.codec, created), http.StatusOK))
}

func deletePod(b *testing.B, bench *podBenchmark) {
	b.Helper()
	pod := bench.pod
	doRequest(b, podRequest{
		ctx:     podRequestContext(pod, "delete", ""),
		target:  fmt.Sprintf("/api/v1/namespaces/%s/pods/%s?gracePeriodSeconds=0", pod.Namespace, pod.Name),
		handler: func() http.HandlerFunc { return handlers.DeleteResource(bench.rest, true, bench.scope, bench.admit) },
	}, http.MethodDelete, "", nil, http.StatusOK)
}

func newRequestScope(b *testing.B, gvk schema.GroupVersionKind, subresource string, resetFields map[fieldpath.APIVersion]*fieldpath.Set) *handlers.RequestScope {
	b.Helper()
	scheme := legacyscheme.Scheme
	fm, err := managedfields.NewDefaultFieldManager(
		applyconfigurations.NewTypeConverter(scheme),
		runtime.UnsafeObjectConvertor(scheme),
		scheme,
		scheme,
		gvk,
		api.SchemeGroupVersion,
		subresource,
		fieldpath.NewExcludeFilterSetMap(resetFields),
	)
	if err != nil {
		b.Fatalf("failed to create field manager: %v", err)
	}
	return &handlers.RequestScope{
		Namer:               handlers.ContextBasedNaming{Namer: meta.NewAccessor()},
		Serializer:          legacyscheme.Codecs,
		ParameterCodec:      legacyscheme.ParameterCodec,
		Creater:             scheme,
		Convertor:           scheme,
		Defaulter:           scheme,
		Typer:               scheme,
		UnsafeConvertor:     runtime.UnsafeObjectConvertor(scheme),
		MaxRequestBodyBytes: 3 * 1024 * 1024,
		Kind:                gvk,
		Resource:            v1.SchemeGroupVersion.WithResource("pods"),
		Subresource:         subresource,
		MetaGroupVersion:    metav1.SchemeGroupVersion,
		HubGroupVersion:     api.SchemeGroupVersion,
		FieldManager:        fm,
	}
}

func setupPodBenchmark(b *testing.B) *podBenchmark {
	b.Helper()
	protobufInfo, ok := runtime.SerializerInfoForMediaType(legacyscheme.Codecs.SupportedMediaTypes(), runtime.ContentTypeProtobuf)
	if !ok {
		b.Fatal("protobuf serializer not found")
	}
	storageCodec := legacyscheme.Codecs.CodecForVersions(
		protobufInfo.Serializer,
		legacyscheme.Codecs.UniversalDeserializer(),
		v1.SchemeGroupVersion,
		runtime.InternalGroupVersioner,
	)
	etcdConfig := framework.SharedEtcd()
	etcdConfig.Codec = storageCodec
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdConfig.ForResource(schema.GroupResource{Resource: "pods"}),
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "pods",
	}
	podStorage, err := podstore.NewStorage(restOptions, nil, nil, nil, nil)
	if err != nil {
		b.Fatalf("unexpected error from REST storage: %v", err)
	}
	b.Cleanup(podStorage.Pod.Destroy)

	gvk := v1.SchemeGroupVersion.WithKind("Pod")
	scope := newRequestScope(b, gvk, "", podStorage.Pod.GetResetFields())
	statusScope := newRequestScope(b, gvk, "status", podStorage.Status.GetResetFields())

	bench := &podBenchmark{
		rest:        podStorage.Pod,
		statusREST:  podStorage.Status,
		scope:       scope,
		statusScope: statusScope,
		admit:       admission.NewChainHandler(admission.NewHandler(admission.Create, admission.Update, admission.Delete)),
		codec:       storageCodec,
	}
	bench.pod = seedPod(b, bench)
	return bench
}
