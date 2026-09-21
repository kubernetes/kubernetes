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
	"fmt"
	"reflect"
	"slices"
	"testing"

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
