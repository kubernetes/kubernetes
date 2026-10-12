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
	_ "embed"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	"k8s.io/apimachinery/pkg/util/yaml"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

const (
	acceptJSON     = kruntime.ContentTypeJSON
	acceptProtobuf = kruntime.ContentTypeProtobuf
	acceptCBOR     = kruntime.ContentTypeCBOR
	acceptYAML     = kruntime.ContentTypeYAML
	acceptTable    = "application/json;as=Table;g=meta.k8s.io;v=v1"
)

type contentType struct {
	name   string
	accept string
}

var allReadFormats = []contentType{
	{name: "json", accept: acceptJSON},
	{name: "protobuf", accept: acceptProtobuf},
	{name: "cbor", accept: acceptCBOR},
	{name: "yaml", accept: acceptYAML},
	{name: "table", accept: acceptTable},
}

var watchFormats = []contentType{
	{name: "json", accept: acceptJSON},
	{name: "protobuf", accept: acceptProtobuf},
	{name: "cbor", accept: acceptCBOR},
	{name: "table", accept: acceptTable},
}

var writeFormats = []contentType{
	{name: "json", accept: acceptJSON},
	{name: "protobuf", accept: acceptProtobuf},
	{name: "cbor", accept: acceptCBOR},
}

//go:embed testdata/exemplar_pod.yaml
var exemplarPodYAML []byte

var exemplarPod = mustLoadExemplarPod()

var podCodecs = serializer.NewCodecFactory(
	legacyscheme.Scheme,
	serializer.WithSerializer(cbor.NewSerializerInfo),
)

func mustLoadExemplarPod() *v1.Pod {
	var pod v1.Pod
	if err := yaml.Unmarshal(exemplarPodYAML, &pod); err != nil {
		panic(fmt.Sprintf("failed to unmarshal exemplar_pod.yaml: %v", err))
	}
	pod.UID = ""
	pod.ResourceVersion = ""
	pod.CreationTimestamp = metav1.Time{}
	pod.ManagedFields = nil
	pod.Spec.NodeName = ""
	return &pod
}

func encodeObject(tb testing.TB, mediaType string, obj kruntime.Object) []byte {
	tb.Helper()
	if mediaType == "" || mediaType == acceptTable {
		mediaType = acceptJSON
	}
	info, ok := kruntime.SerializerInfoForMediaType(podCodecs.SupportedMediaTypes(), mediaType)
	if !ok {
		tb.Fatalf("serializer not found for %s", mediaType)
	}
	codec := podCodecs.CodecForVersions(info.Serializer, podCodecs.UniversalDeserializer(), v1.SchemeGroupVersion, v1.SchemeGroupVersion)
	data, err := kruntime.Encode(codec, obj)
	if err != nil {
		tb.Fatalf("failed to encode %T as %s: %v", obj, mediaType, err)
	}
	return data
}

func ensureDefaultServiceAccount(tb testing.TB, client clientset.Interface) {
	tb.Helper()
	const ns = metav1.NamespaceDefault
	if _, err := client.CoreV1().ServiceAccounts(ns).Create(tb.Context(), &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: ns},
	}, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		tb.Fatalf("failed to create default ServiceAccount in %s: %v", ns, err)
	}
}

// seedExemplarPod creates a pod and then updates its status subresource.
// kube-apiserver's podStrategy.PrepareForCreate resets pod.Status on creation,
// so a second write to /status is required to populate the exemplar's status
// payload and kubelet managedFields entry (~10KB total stored size).
func seedExemplarPod(tb testing.TB, client clientset.Interface, name string) *v1.Pod {
	tb.Helper()
	const ns = metav1.NamespaceDefault
	ctx := tb.Context()
	pod := exemplarPod.DeepCopy()
	pod.Namespace = ns
	pod.Name = name

	created, err := client.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{FieldManager: "seed-client"})
	if err != nil {
		tb.Fatalf("seed create %s/%s failed: %v", ns, name, err)
	}
	created.Status = exemplarPod.Status
	updated, err := client.CoreV1().Pods(ns).UpdateStatus(ctx, created, metav1.UpdateOptions{FieldManager: "kubelet"})
	if err != nil {
		tb.Fatalf("seed status update %s/%s failed: %v", ns, name, err)
	}
	return updated
}

func seedPods(tb testing.TB, client clientset.Interface, count int) []*v1.Pod {
	tb.Helper()
	pods := make([]*v1.Pod, count)
	for i := range count {
		pods[i] = seedExemplarPod(tb, client, fmt.Sprintf("pod-%03d", i))
	}
	if _, err := client.CoreV1().Pods(metav1.NamespaceDefault).List(tb.Context(), metav1.ListOptions{}); err != nil {
		tb.Fatalf("waiting for watch cache sync failed: %v", err)
	}
	return pods
}

func staticPodBody(tb testing.TB, name string) func(mediaType string, _ *v1.Pod) []byte {
	tb.Helper()
	pod := exemplarPod.DeepCopy()
	pod.Namespace = metav1.NamespaceDefault
	pod.Name = name
	return staticObjectBody(tb, pod)
}

func staticObjectBody(tb testing.TB, obj kruntime.Object) func(mediaType string, _ *v1.Pod) []byte {
	tb.Helper()
	encodedByType := make(map[string][]byte, len(writeFormats))
	for _, wf := range writeFormats {
		encodedByType[wf.accept] = encodeObject(tb, wf.accept, obj)
	}
	return func(mediaType string, _ *v1.Pod) []byte {
		return encodedByType[mediaType]
	}
}

func staticBytesBody(payload string) func(string, *v1.Pod) []byte {
	b := []byte(payload)
	return func(string, *v1.Pod) []byte {
		return b
	}
}
