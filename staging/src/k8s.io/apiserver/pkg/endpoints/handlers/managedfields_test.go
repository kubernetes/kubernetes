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

package handlers

import (
	"bytes"
	_ "embed"
	"fmt"
	"io"
	"reflect"
	"slices"
	"testing"

	"sigs.k8s.io/yaml"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
)

func newPodWithManagedFields(name string) *examplev1.Pod {
	grace := int64(30)
	return &examplev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: examplev1.SchemeGroupVersion.String(), Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
			Labels:    map[string]string{"app": name},
			ManagedFields: []metav1.ManagedFieldsEntry{
				{Manager: "kube-controller-manager", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", FieldsType: "FieldsV1", FieldsV1: metav1.NewFieldsV1(`{"f:metadata":{"f:labels":{"f:app":{}}}}`)},
				{Manager: "kubelet", Operation: metav1.ManagedFieldsOperationUpdate, APIVersion: "v1", Subresource: "status", FieldsType: "FieldsV1", FieldsV1: metav1.NewFieldsV1(`{"f:status":{"f:phase":{}}}`)},
			},
		},
		Spec: examplev1.PodSpec{
			NodeSelector:                  map[string]string{"disktype": "ssd"},
			TerminationGracePeriodSeconds: &grace,
		},
		Status: examplev1.PodStatus{
			Phase:      "Running",
			Conditions: []examplev1.PodCondition{{Type: "Ready", Status: "True"}},
		},
	}
}

func newPodListWithManagedFields(n int) *examplev1.PodList {
	list := &examplev1.PodList{
		TypeMeta: metav1.TypeMeta{APIVersion: examplev1.SchemeGroupVersion.String(), Kind: "PodList"},
		ListMeta: metav1.ListMeta{ResourceVersion: "12345"},
	}
	for i := range n {
		list.Items = append(list.Items, *newPodWithManagedFields(fmt.Sprintf("pod-%d", i)))
	}
	return list
}

func TestDropManagedFields(t *testing.T) {
	orig := newPodWithManagedFields("a")
	stripped := dropManagedFields(orig).(*examplev1.Pod)

	if len(stripped.ManagedFields) != 0 {
		t.Errorf("copy still has %d managedFields entries", len(stripped.ManagedFields))
	}
	if len(orig.ManagedFields) != 2 {
		t.Errorf("original mutated: %d managedFields entries, want 2", len(orig.ManagedFields))
	}
	// Interior memory is shared, not copied.
	if &orig.Status.Conditions[0] != &stripped.Status.Conditions[0] {
		t.Error("conditions backing array was copied, want shared")
	}
	if reflect.ValueOf(orig.Labels).Pointer() != reflect.ValueOf(stripped.Labels).Pointer() {
		t.Error("labels map was copied, want shared")
	}
	if orig.Spec.TerminationGracePeriodSeconds != stripped.Spec.TerminationGracePeriodSeconds {
		t.Error("spec pointer field was copied, want shared")
	}
}

func TestDropManagedFieldsList(t *testing.T) {
	cached := newPodListWithManagedFields(3)
	// The cacher copies cached objects into a fresh Items slice for each list.
	list := &examplev1.PodList{Items: slices.Clone(cached.Items)}
	if dropManagedFields(list) != list {
		t.Error("list was copied, want cleared in place")
	}
	for i := range list.Items {
		if len(list.Items[i].ManagedFields) != 0 {
			t.Errorf("item %d still has managedFields", i)
		}
		if len(cached.Items[i].ManagedFields) != 2 {
			t.Errorf("cached item %d mutated", i)
		}
	}
}

func TestDropManagedFieldsUnstructured(t *testing.T) {
	content := func(name string) map[string]interface{} {
		return map[string]interface{}{
			"apiVersion": "example.com/v1",
			"kind":       "Widget",
			"metadata": map[string]interface{}{
				"name":          name,
				"managedFields": []interface{}{map[string]interface{}{"manager": "m"}},
			},
			"spec": map[string]interface{}{"replicas": int64(3)},
		}
	}
	hasManagedFields := func(u *unstructured.Unstructured) bool {
		_, found := u.Object["metadata"].(map[string]interface{})["managedFields"]
		return found
	}

	orig := &unstructured.Unstructured{Object: content("w")}
	stripped := dropManagedFields(orig).(*unstructured.Unstructured)
	if hasManagedFields(stripped) {
		t.Error("copy still has managedFields")
	}
	if !hasManagedFields(orig) {
		t.Error("original mutated: managedFields removed from shared map")
	}
	if reflect.ValueOf(orig.Object["spec"]).Pointer() != reflect.ValueOf(stripped.Object["spec"]).Pointer() {
		t.Error("spec map was copied, want shared")
	}

	cached := &unstructured.Unstructured{Object: content("w")}
	list := &unstructured.UnstructuredList{Items: []unstructured.Unstructured{*cached}}
	if dropManagedFields(list) != list {
		t.Error("list was copied, want cleared in place")
	}
	if hasManagedFields(&list.Items[0]) {
		t.Error("list item still has managedFields")
	}
	if !hasManagedFields(cached) {
		t.Error("cached item mutated: managedFields removed from shared map")
	}
}

func TestDropManagedFieldsUnchanged(t *testing.T) {
	pod := newPodWithManagedFields("a")
	pod.ManagedFields = nil
	for _, obj := range []runtime.Object{
		&metav1.Status{Status: metav1.StatusFailure, Code: 404},
		pod,
	} {
		if dropManagedFields(obj) != obj {
			t.Errorf("%T: expected the object to be returned unchanged", obj)
		}
	}
}

// pointerMetaObject shares its metadata with any shallow copy.
type pointerMetaObject struct {
	metav1.TypeMeta
	*metav1.ObjectMeta
}

func (o *pointerMetaObject) DeepCopyObject() runtime.Object {
	return &pointerMetaObject{TypeMeta: o.TypeMeta, ObjectMeta: o.ObjectMeta.DeepCopy()}
}

func TestDropManagedFieldsPointerMetadata(t *testing.T) {
	orig := &pointerMetaObject{ObjectMeta: &metav1.ObjectMeta{Name: "a", ManagedFields: []metav1.ManagedFieldsEntry{{Manager: "m"}}}}
	if len(dropManagedFields(orig).(*pointerMetaObject).ManagedFields) != 0 {
		t.Error("copy still has managedFields")
	}
	if len(orig.ManagedFields) != 1 {
		t.Error("original mutated through shared metadata")
	}
}

// TestDropManagedFieldsEncoding checks the result encodes exactly like a deep
// copy with managedFields cleared, for every serializer.
func TestDropManagedFieldsEncoding(t *testing.T) {
	encoders := map[string]runtime.Encoder{
		"json":     json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{}),
		"protobuf": protobuf.NewSerializer(nil, nil),
		"cbor":     cbor.NewSerializer(nil, nil),
	}
	for name, encoder := range encoders {
		t.Run(name, func(t *testing.T) {
			for _, obj := range []runtime.Object{newPodWithManagedFields("a"), newPodListWithManagedFields(3)} {
				var got, expected bytes.Buffer
				// Lists are cleared in place, so encode the deep copy first.
				if err := encoder.Encode(deepCopyWithoutManagedFields(obj), &expected); err != nil {
					t.Fatal(err)
				}
				if err := encoder.Encode(dropManagedFields(obj), &got); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got.Bytes(), expected.Bytes()) {
					t.Errorf("%T: encoding differs from deep copy encoding\ngot:  %q\nwant: %q", obj, got.String(), expected.String())
				}
				if bytes.Contains(got.Bytes(), []byte("kubelet")) {
					t.Errorf("%T: encoding still contains a managedFields manager", obj)
				}
			}
		})
	}
}

//go:embed responsewriters/testdata/exemplar_pod.yaml
var benchmarkExemplarPodYAML []byte

// BenchmarkDropManagedFields encodes an exemplar Pod and a LIST of 1000 of them,
// typed and unstructured, with and without managedFields, using the encoders
// the apiserver serves them with.
func BenchmarkDropManagedFields(b *testing.B) {
	var pod corev1.Pod
	if err := yaml.Unmarshal(benchmarkExemplarPodYAML, &pod); err != nil {
		b.Fatal(err)
	}
	content, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&pod)
	if err != nil {
		b.Fatal(err)
	}
	cached := slices.Repeat([]corev1.Pod{pod}, 1000)
	cachedUnstructured := slices.Repeat([]unstructured.Unstructured{{Object: content}}, 1000)
	objects := []struct {
		name   string
		object func() runtime.Object
	}{
		{"Pod", func() runtime.Object { return &pod }},
		// The cacher copies cached objects into a fresh list for each request.
		{"PodList", func() runtime.Object { return &corev1.PodList{Items: slices.Clone(cached)} }},
		{"Unstructured", func() runtime.Object { return &unstructured.Unstructured{Object: content} }},
		{"UnstructuredList", func() runtime.Object {
			return &unstructured.UnstructuredList{Items: slices.Clone(cachedUnstructured)}
		}},
	}
	encoders := []struct {
		name    string
		encoder runtime.Encoder
	}{
		{"Json", json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{StreamingCollectionsEncoding: true})},
		// The apiserver encodes protobuf with a pooled allocator.
		{"Protobuf", runtime.NewEncoderWithAllocator(protobuf.NewSerializerWithOptions(nil, nil, protobuf.SerializerOptions{StreamingCollectionsEncoding: true}), &runtime.Allocator{})},
		{"Cbor", cbor.NewSerializer(nil, nil)},
	}
	for _, o := range objects {
		for _, e := range encoders {
			if _, ok := o.object().(runtime.Unstructured); ok && e.name == "Protobuf" {
				continue // Protobuf can't encode unstructured objects.
			}
			for _, drop := range []bool{false, true} {
				b.Run(fmt.Sprintf("Object=%s/MediaType=%s/Drop=%t", o.name, e.name, drop), func(b *testing.B) {
					encode := func(w io.Writer) {
						obj := o.object()
						if drop {
							obj = dropManagedFields(obj)
						}
						if err := e.encoder.Encode(obj, w); err != nil {
							b.Fatal(err)
						}
					}
					var written bytes.Buffer
					encode(&written)
					b.ReportAllocs()
					for b.Loop() {
						encode(io.Discard)
					}
					b.ReportMetric(float64(written.Len()), "writtenBytes/op")
				})
			}
		}
	}
}
