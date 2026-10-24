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

package unstructured

import (
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utiljson "k8s.io/apimachinery/pkg/util/json"
)

func TestObjectGetSetDelete(t *testing.T) {
	var o Object
	if v, ok := o.Get("a"); ok || v != nil {
		t.Fatalf("unexpected result for empty object: %v, %v", v, ok)
	}
	o.Set("b", 1)
	o.Set("a", nil)
	o.Set("b", 2)
	if want := (Object{{"b", 2}, {"a", nil}}); !reflect.DeepEqual(o, want) {
		t.Fatalf("got %v, want %v", o, want)
	}
	if v, ok := o.Get("a"); !ok || v != nil {
		t.Fatalf("explicit nil must be found, got %v, %v", v, ok)
	}
	o.Delete("b")
	o.Delete("missing")
	if want := (Object{{"a", nil}}); !reflect.DeepEqual(o, want) {
		t.Fatalf("got %v, want %v", o, want)
	}
}

// Duplicate fields are invalid JSON, but the existing decoders accept them
// and the last one wins. StableUnstructured must behave the same.
func TestDuplicateFieldsLastWins(t *testing.T) {
	const data = `{"a":1,"b":2,"a":3}`

	var std map[string]any
	if err := json.Unmarshal(data2bytes(data), &std); err != nil {
		t.Fatal(err)
	}
	var k8s map[string]any
	if err := utiljson.Unmarshal(data2bytes(data), &k8s); err != nil {
		t.Fatal(err)
	}
	if std["a"] != float64(3) || k8s["a"] != int64(3) {
		t.Fatalf("assumption broken: encoding/json=%v, k8s json=%v", std["a"], k8s["a"])
	}

	var o Object
	if err := json.Unmarshal(data2bytes(data), &o); err != nil {
		t.Fatal(err)
	}
	if v, _ := o.Get("a"); v != int64(3) {
		t.Fatalf("got %v, want 3", v)
	}
	o.Set("a", int64(4))
	if v, _ := o.Get("a"); v != int64(4) {
		t.Fatalf("got %v, want 4", v)
	}
}

func data2bytes(s string) []byte { return []byte(s) }

func TestStableRoundTrip(t *testing.T) {
	tests := map[string]struct {
		in      string
		wantErr bool
	}{
		"ordered":       {in: `{"z":1,"a":{"y":[1,2,{"q":null,"b":true}],"c":"x"},"m":1.5}`},
		"empty":         {in: `{}`},
		"escaped":       {in: `{"k\"ey":"v\u003cal"}`},
		"not-object":    {in: `[1]`, wantErr: true},
		"truncated":     {in: `{"a":`, wantErr: true},
		"trailing":      {in: `{"a":1} x`, wantErr: true},
		"missing-comma": {in: `{"a":1 "b":2}`, wantErr: true},
	}
	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			var o Object
			err := json.Unmarshal([]byte(tc.in), &o)
			if (err != nil) != tc.wantErr {
				t.Fatalf("unexpected error state: %v", err)
			}
			if err != nil {
				return
			}
			out, err := json.Marshal(o)
			if err != nil {
				t.Fatal(err)
			}
			if string(out) != tc.in {
				t.Fatalf("got %s, want %s", out, tc.in)
			}
		})
	}
}

func TestStableNumbers(t *testing.T) {
	var o Object
	if err := json.Unmarshal([]byte(`{"i":1,"f":1.5,"w":2.0,"big":1e30}`), &o); err != nil {
		t.Fatal(err)
	}
	want := Object{{"i", int64(1)}, {"f", 1.5}, {"w", int64(2)}, {"big", 1e30}}
	if !reflect.DeepEqual(o, want) {
		t.Fatalf("got %#v, want %#v", o, want)
	}
}

func TestStableUnstructuredCodec(t *testing.T) {
	const data = `{"spec":{"b":1,"a":2},"kind":"Pod","apiVersion":"v1","metadata":{"name":"x","namespace":"ns"}}`
	obj, gvk, err := StableUnstructuredJSONScheme.Decode([]byte(data), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gvk.Kind != "Pod" || gvk.Version != "v1" {
		t.Fatalf("unexpected gvk %v", gvk)
	}
	u := obj.(*StableUnstructured)
	if u.GetName() != "x" || u.GetNamespace() != "ns" {
		t.Fatalf("unexpected accessors: %q %q", u.GetName(), u.GetNamespace())
	}
	out, err := runtime.Encode(StableUnstructuredJSONScheme, u)
	if err != nil {
		t.Fatal(err)
	}
	if string(out) != data+"\n" {
		t.Fatalf("got %s", out)
	}

	if _, _, err := StableUnstructuredJSONScheme.Decode([]byte(`{"a":1}`), nil, nil); !runtime.IsMissingKind(err) {
		t.Fatalf("expected missing kind error, got %v", err)
	}
}

func TestStableUnstructuredAccessors(t *testing.T) {
	u := &StableUnstructured{}
	u.SetGroupVersionKind(schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"})
	u.SetName("n")
	u.SetNamespace("ns")
	u.SetLabels(map[string]string{"b": "1", "a": "2"})
	u.SetResourceVersion("5")
	u.SetNamespace("")
	u.SetResourceVersion("")

	if u.GetName() != "n" || u.GetNamespace() != "" || u.GetResourceVersion() != "" {
		t.Fatalf("unexpected state: %v", u.Object)
	}
	if got := u.GetLabels(); !reflect.DeepEqual(got, map[string]string{"a": "2", "b": "1"}) {
		t.Fatalf("unexpected labels %v", got)
	}
	out, err := json.Marshal(u)
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"apiVersion":"apps/v1","kind":"Deployment","metadata":{"name":"n","labels":{"a":"2","b":"1"}}}`
	if string(out) != want {
		t.Fatalf("got %s, want %s", out, want)
	}
}

func TestStableUnstructuredList(t *testing.T) {
	const data = `{"kind":"PodList","apiVersion":"v1","metadata":{"resourceVersion":"1"},"items":[{"metadata":{"name":"a"},"spec":{}},{"kind":"Pod","apiVersion":"v1","metadata":{"name":"b"}}]}`
	obj, _, err := StableUnstructuredJSONScheme.Decode([]byte(data), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	u := obj.(*StableUnstructured)
	if !u.IsList() {
		t.Fatal("expected a list")
	}
	var names []string
	err = u.EachListItem(func(o runtime.Object) error {
		item := o.(*StableUnstructured)
		if item.GetKind() != "Pod" || item.GetAPIVersion() != "v1" {
			t.Errorf("item %q has gvk %v", item.GetName(), item.GroupVersionKind())
		}
		names = append(names, item.GetName())
		item.SetUID("u") // must be visible in the list
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(names, []string{"a", "b"}) {
		t.Fatalf("got %v", names)
	}
	out, _ := json.Marshal(u)
	const want = `{"kind":"PodList","apiVersion":"v1","metadata":{"resourceVersion":"1"},"items":[{"kind":"Pod","apiVersion":"v1","metadata":{"name":"a","uid":"u"},"spec":{}},{"kind":"Pod","apiVersion":"v1","metadata":{"name":"b","uid":"u"}}]}`
	if string(out) != want {
		t.Fatalf("got %s\nwant %s", out, want)
	}
}

func TestStableUnstructuredDeepCopy(t *testing.T) {
	u := &StableUnstructured{}
	if err := json.Unmarshal([]byte(`{"kind":"A","apiVersion":"v1","l":[{"x":1}]}`), u); err != nil {
		t.Fatal(err)
	}
	c := u.DeepCopy()
	inner := c.Object[2].Value.([]any)[0].(Object)
	inner.Set("x", int64(2))
	c.Object[2].Value.([]any)[0] = inner
	if v, _ := u.Object[2].Value.([]any)[0].(Object).Get("x"); v != int64(1) {
		t.Fatalf("original modified: %v", v)
	}
}
