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
	"net/http"
	"net/http/httptest"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/client-go/applyconfigurations"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"sigs.k8s.io/structured-merge-diff/v7/fieldpath"
)

//go:embed testdata/exemplar_pod.yaml
var exemplarPodYAML []byte

func BenchmarkPatchPod(b *testing.B) {
	podStorage, _, scope, admit, pod := setupBenchmarkPatch(b, "")
	patchTypes := []string{string(types.StrategicMergePatchType)}
	ctx := audit.WithAuditContext(request.WithNamespace(request.WithRequestInfo(context.Background(), &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "patch",
		APIVersion:        "v1",
		Resource:          "pods",
		Namespace:         pod.Namespace,
		Name:              pod.Name,
	}), pod.Namespace))
	target := fmt.Sprintf("/api/v1/namespaces/%s/pods/%s?fieldManager=patch-manager", pod.Namespace, pod.Name)

	patch := func(i int) {
		body := bytes.NewReader(fmt.Appendf(nil, `{"metadata":{"labels":{"bench-updated":"%d"}}}`, i))
		req := httptest.NewRequestWithContext(ctx, http.MethodPatch, target, body)
		req.Header.Set("Content-Type", string(types.StrategicMergePatchType))
		req.Header.Set("Accept", runtime.ContentTypeProtobuf)
		w := httptest.NewRecorder()
		handlers.PatchResource(podStorage, scope, admit, patchTypes)(w, req)
		if w.Code != http.StatusOK {
			b.Fatalf("unexpected status %d: %s", w.Code, w.Body.String())
		}
	}
	patch(0)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		patch(i + 1)
	}
}

func BenchmarkPatchPodStatus(b *testing.B) {
	_, statusStorage, scope, admit, pod := setupBenchmarkPatch(b, "status")
	patchTypes := []string{string(types.StrategicMergePatchType)}
	ctx := audit.WithAuditContext(request.WithNamespace(request.WithRequestInfo(context.Background(), &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "patch",
		APIVersion:        "v1",
		Resource:          "pods",
		Subresource:       "status",
		Namespace:         pod.Namespace,
		Name:              pod.Name,
	}), pod.Namespace))
	target := fmt.Sprintf("/api/v1/namespaces/%s/pods/%s/status?fieldManager=kubelet", pod.Namespace, pod.Name)

	patch := func(i int) {
		readyStatus := "True"
		if i%2 == 0 {
			readyStatus = "False"
		}
		body := bytes.NewReader(fmt.Appendf(nil,
			`{"status":{"conditions":[{"type":"Ready","status":%q,"message":"kubelet-probe-%d"},{"type":"ContainersReady","status":%q}]}}`,
			readyStatus, i, readyStatus))
		req := httptest.NewRequestWithContext(ctx, http.MethodPatch, target, body)
		req.Header.Set("Content-Type", string(types.StrategicMergePatchType))
		req.Header.Set("Accept", runtime.ContentTypeProtobuf)
		w := httptest.NewRecorder()
		handlers.PatchResource(statusStorage, scope, admit, patchTypes)(w, req)
		if w.Code != http.StatusOK {
			b.Fatalf("unexpected status %d: %s", w.Code, w.Body.String())
		}
	}
	patch(0)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		patch(i + 1)
	}
}

func setupBenchmarkPatch(b *testing.B, subresource string) (*REST, *StatusREST, *handlers.RequestScope, admission.Interface, *api.Pod) {
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
	store := &codecStorage{
		codec:     storageCodec,
		versioner: storage.APIObjectVersioner{},
		rev:       1,
	}
	restOptions := generic.RESTOptions{
		StorageConfig: &storagebackend.ConfigForResource{
			Config:        storagebackend.Config{Prefix: "/registry", Codec: storageCodec},
			GroupResource: schema.GroupResource{Resource: "pods"},
		},
		Decorator: func(*storagebackend.ConfigForResource, string, func(runtime.Object) (string, error), func() runtime.Object, func() runtime.Object, storage.AttrFunc, storage.IndexerFuncs, *cache.Indexers) (storage.Interface, factory.DestroyFunc, error) {
			return store, func() {}, nil
		},
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "pods",
	}
	podStorage, err := NewStorage(restOptions, nil, nil, nil, nil)
	if err != nil {
		b.Fatalf("unexpected error from REST storage: %v", err)
	}

	resetFields := podStorage.Pod.GetResetFields()
	if subresource == "status" {
		resetFields = podStorage.Status.GetResetFields()
	}

	scheme := legacyscheme.Scheme
	gvk := v1.SchemeGroupVersion.WithKind("Pod")
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

	pod := createBenchmarkPod(b, podStorage.Pod, podStorage.Status)
	scope := &handlers.RequestScope{
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
	admit := admission.NewChainHandler(admission.NewHandler(admission.Update))
	return podStorage.Pod, podStorage.Status, scope, admit, pod
}

func createBenchmarkPod(b *testing.B, storage *REST, statusStorage *StatusREST) *api.Pod {
	b.Helper()
	var v1Pod v1.Pod
	if err := yaml.Unmarshal(exemplarPodYAML, &v1Pod); err != nil {
		b.Fatalf("failed to unmarshal exemplar_pod.yaml: %v", err)
	}
	var internalPod api.Pod
	if err := legacyscheme.Scheme.Convert(&v1Pod, &internalPod, nil); err != nil {
		b.Fatalf("failed to convert pod: %v", err)
	}
	desiredStatus := *internalPod.Status.DeepCopy()
	internalPod.ResourceVersion = ""
	ctx := request.WithNamespace(request.WithRequestInfo(context.Background(), &request.RequestInfo{
		IsResourceRequest: true,
		Verb:              "create",
		APIVersion:        "v1",
		Resource:          "pods",
		Namespace:         internalPod.Namespace,
	}), internalPod.Namespace)
	created, err := storage.Create(ctx, &internalPod, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		b.Fatalf("failed to create pod: %v", err)
	}
	createdPod := created.(*api.Pod)
	createdPod.Status = desiredStatus
	updated, _, err := statusStorage.Update(ctx, createdPod.Name, rest.DefaultUpdatedObjectInfo(createdPod), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		b.Fatalf("failed to update pod status: %v", err)
	}
	return updated.(*api.Pod)
}

// codecStorage implements storage.Interface using the Protobuf storage codec
// (matching etcd3.store's Encode/Decode round-trip) without an in-process embedded
// etcd server polluting b.ReportAllocs().
type codecStorage struct {
	codec     runtime.Codec
	versioner storage.Versioner
	rev       uint64
	data      []byte
	cachedObj runtime.Object
}

func (s *codecStorage) Versioner() storage.Versioner { return s.versioner }
func (s *codecStorage) CompactRevision() int64       { return 0 }
func (s *codecStorage) ReadinessCheck() error        { return nil }
func (s *codecStorage) EnableResourceSizeEstimation(storage.KeysFunc) error {
	return nil
}
func (s *codecStorage) RequestWatchProgress(context.Context) error { return nil }
func (s *codecStorage) GetCurrentResourceVersion(context.Context) (uint64, error) {
	return s.rev, nil
}
func (s *codecStorage) Stats(context.Context) (storage.Stats, error) {
	return storage.Stats{}, nil
}
func (s *codecStorage) Watch(context.Context, string, storage.ListOptions) (watch.Interface, error) {
	return nil, nil
}
func (s *codecStorage) GetList(context.Context, string, storage.ListOptions, runtime.Object) error {
	return nil
}
func (s *codecStorage) Delete(context.Context, string, runtime.Object, *storage.Preconditions, storage.ValidateObjectFunc, runtime.Object, storage.DeleteOptions) error {
	return nil
}
func (s *codecStorage) Get(_ context.Context, _ string, _ storage.GetOptions, objPtr runtime.Object) error {
	if err := runtime.DecodeInto(s.codec, s.data, objPtr); err != nil {
		return err
	}
	return s.versioner.UpdateObject(objPtr, s.rev)
}
func (s *codecStorage) Create(_ context.Context, _ string, obj, out runtime.Object, _ uint64) error {
	if err := s.versioner.PrepareObjectForStorage(obj); err != nil {
		return err
	}
	data, err := runtime.Encode(s.codec, obj)
	if err != nil {
		return err
	}
	s.rev++
	s.data = data
	s.cachedObj = obj.DeepCopyObject()
	_ = s.versioner.UpdateObject(s.cachedObj, s.rev)
	if out != nil {
		if err := runtime.DecodeInto(s.codec, data, out); err != nil {
			return err
		}
		return s.versioner.UpdateObject(out, s.rev)
	}
	return nil
}
func (s *codecStorage) GuaranteedUpdate(_ context.Context, key string, destination runtime.Object, _ bool, preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, _ runtime.Object) error {
	curr := s.cachedObj.DeepCopyObject()
	if preconditions != nil {
		if err := preconditions.Check(key, curr); err != nil {
			return err
		}
	}
	ret, _, err := tryUpdate(curr, storage.ResponseMeta{ResourceVersion: s.rev})
	if err != nil {
		return err
	}
	if err := s.versioner.PrepareObjectForStorage(ret); err != nil {
		return err
	}
	data, err := runtime.Encode(s.codec, ret)
	if err != nil {
		return err
	}
	s.rev++
	s.data = data
	if err := runtime.DecodeInto(s.codec, data, destination); err != nil {
		return err
	}
	if err := s.versioner.UpdateObject(destination, s.rev); err != nil {
		return err
	}
	s.cachedObj = destination.DeepCopyObject()
	return nil
}
