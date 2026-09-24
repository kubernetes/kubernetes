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
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/warning"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/structured-merge-diff/v7/fieldpath"
)

func unprunedStrategicPatchObject(ctx context.Context, defaulter runtime.ObjectDefaulter, originalObject runtime.Object, patchBytes []byte, objToUpdate, schemaReferenceObj runtime.Object, validationDirective string) error {
	originalObjMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(originalObject)
	if err != nil {
		return err
	}
	patchMap, strictErrs, err := decodePatchMap(patchBytes, validationDirective)
	if err != nil {
		return err
	}
	return applyPatchToObject(ctx, defaulter, originalObjMap, patchMap, objToUpdate, schemaReferenceObj, strictErrs, validationDirective, nil, nil, nil)
}

type recordingWarnings struct{ warnings []string }

func (r *recordingWarnings) AddWarning(_, text string) { r.warnings = append(r.warnings, text) }

type patchOutcome struct {
	obj      runtime.Object
	err      string
	warnings []string
}

func runPatch(t *testing.T, pruned bool, original runtime.Object, patch []byte, directive string) patchOutcome {
	t.Helper()
	recorder := &recordingWarnings{}
	ctx := warning.WithWarningRecorder(request.WithNamespace(context.Background(), "ns"), recorder)
	originalCopy := original.DeepCopyObject()
	obj := reflect.New(reflect.TypeOf(original).Elem()).Interface().(runtime.Object)
	schemaReferenceObj := reflect.New(reflect.TypeOf(original).Elem()).Interface().(runtime.Object)
	var err error
	if pruned {
		err = strategicPatchObject(ctx, clientgoscheme.Scheme, original, patch, obj, schemaReferenceObj, directive)
	} else {
		err = unprunedStrategicPatchObject(ctx, clientgoscheme.Scheme, original, patch, obj, schemaReferenceObj, directive)
	}
	if !reflect.DeepEqual(originalCopy, original) {
		t.Fatalf("patch mutated the original: %s", cmp.Diff(originalCopy, original))
	}
	out := patchOutcome{obj: obj, warnings: recorder.warnings}
	if err != nil {
		out.err = err.Error()
		out.obj = nil
	}
	sort.Strings(out.warnings)
	return out
}

func comparePatchOutcomes(t *testing.T, original runtime.Object, patch []byte, directive string) (pruned bool) {
	t.Helper()
	patchMap, _, err := decodePatchMap(patch, directive)
	if err == nil {
		pruned = canPruneTopLevelPatch(patchMap)
	}
	want := runPatch(t, false, original, patch, directive)
	got := runPatch(t, true, original, patch, directive)
	if want.err != got.err {
		t.Fatalf("patch %s: error mismatch\nunpruned: %v\npruned:   %v", patch, want.err, got.err)
	}
	if !reflect.DeepEqual(want.warnings, got.warnings) {
		t.Fatalf("patch %s: warnings mismatch\nunpruned: %v\npruned:   %v", patch, want.warnings, got.warnings)
	}
	if want.obj == nil {
		return pruned
	}
	canonicalizeFieldsV1(t, want.obj)
	canonicalizeFieldsV1(t, got.obj)
	if !apiequality.Semantic.DeepEqual(want.obj, got.obj) {
		t.Fatalf("patch %s: result mismatch (-unpruned +pruned):\n%s", patch, cmp.Diff(want.obj, got.obj))
	}
	wantJSON, err := json.Marshal(want.obj)
	if err != nil {
		t.Fatal(err)
	}
	gotJSON, err := json.Marshal(got.obj)
	if err != nil {
		t.Fatal(err)
	}
	if string(wantJSON) != string(gotJSON) {
		t.Fatalf("patch %s: serialized result mismatch\nunpruned: %s\npruned:   %s", patch, wantJSON, gotJSON)
	}
	originalCopy := original.DeepCopyObject()
	scribble(reflect.ValueOf(got.obj))
	if !reflect.DeepEqual(originalCopy, original) {
		t.Fatalf("patch %s: result aliases the original: %s", patch, cmp.Diff(originalCopy, original))
	}
	return pruned
}

func canonicalizeFieldsV1(t *testing.T, obj runtime.Object) {
	t.Helper()
	accessor, err := meta.Accessor(obj)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range accessor.GetManagedFields() {
		if entry.FieldsV1 == nil {
			continue
		}
		set := &fieldpath.Set{}
		if err := set.FromJSON(bytes.NewReader(entry.FieldsV1.Raw)); err != nil {
			continue
		}
		raw, err := set.ToJSON()
		if err != nil {
			t.Fatal(err)
		}
		entry.FieldsV1.Raw = raw
	}
}

func scribble(v reflect.Value) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface:
		if !v.IsNil() {
			scribble(v.Elem())
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if v.Type().Field(i).IsExported() {
				scribble(v.Field(i))
			}
		}
	case reflect.Slice:
		for i := 0; i < v.Len(); i++ {
			scribble(v.Index(i))
		}
	case reflect.Map:
		for _, k := range v.MapKeys() {
			e := reflect.New(v.Type().Elem()).Elem()
			e.Set(v.MapIndex(k))
			scribble(e)
			v.SetMapIndex(k, e)
		}
	case reflect.String:
		if v.CanSet() {
			v.SetString("scribbled")
		}
	case reflect.Int, reflect.Int32, reflect.Int64:
		if v.CanSet() {
			v.SetInt(-7)
		}
	}
}

func testPod() *corev1.Pod {
	return &corev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:        "p",
			Namespace:   "ns",
			Labels:      map[string]string{"a": "b"},
			Annotations: map[string]string{"x": "y"},
			Finalizers:  []string{"f1", "f2"},
			ManagedFields: []metav1.ManagedFieldsEntry{{
				Manager:    "m",
				Operation:  metav1.ManagedFieldsOperationUpdate,
				APIVersion: "v1",
				FieldsType: "FieldsV1",
				FieldsV1:   &metav1.FieldsV1{Raw: []byte(`{"f:metadata":{"f:labels":{".":{},"f:a":{}}},"f:spec":{"f:containers":{"k:{\"name\":\"c\"}":{".":{},"f:image":{}}}}}`)},
			}},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "c", Image: "i", Env: []corev1.EnvVar{{Name: "E", Value: "V"}}}},
		},
		Status: corev1.PodStatus{
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}
}

func TestPrunedStrategicPatchMatchesUnpruned(t *testing.T) {
	cases := []struct {
		patch      string
		wantPruned bool
	}{
		{`{"metadata":{"labels":{"a":"c","d":"e"}}}`, true},
		{`{"metadata":{"labels":null}}`, true},
		{`{"metadata":{"annotations":{"x":null}},"status":{"phase":"Failed"}}`, true},
		{`{"status":{"conditions":[{"type":"Ready","status":"False"}]}}`, true},
		{`{"status":{"$setElementOrder/conditions":[{"type":"Ready"}],"conditions":[{"type":"Ready","status":"False"}]}}`, true},
		{`{"spec":{"containers":[{"name":"c","image":"j"}]}}`, true},
		{`{"spec":{"containers":[{"name":"c","$patch":"delete"}]}}`, true},
		{`{"spec":{"$retainKeys":["containers"],"containers":[{"name":"c","image":"j"}]}}`, true},
		{`{"metadata":{"$deleteFromPrimitiveList/finalizers":["f1"]}}`, true},
		{`{"metadata":{"$setElementOrder/finalizers":["f2","f1"],"finalizers":["f2","f1"]}}`, true},
		{`{"metadata":{"$patch":"replace","name":"p"}}`, true},
		{`{"metadata":{"$retainKeys":["name"],"name":"p"}}`, true},
		{`{"metadata":{"managedFields":[{"manager":"other","operation":"Update","apiVersion":"v1","fieldsType":"FieldsV1","fieldsV1":{"f:spec":{}}}]}}`, true},
		{`{"metadata":{"managedFields":null}}`, true},
		{`{"metadata":null}`, true},
		{`{"status":null}`, true},
		{`{"$setElementOrder/metadata":[]}`, false},
		{`{"apiVersion":"v1","kind":"Pod"}`, true},
		{`{"kind":"Other"}`, true},
		{`{"$patch":"replace","metadata":{"name":"p"}}`, false},
		{`{"$retainKeys":["metadata"],"metadata":{"name":"p"}}`, false},
		{`{"unknown":1,"metadata":{"labels":{"a":"z"}}}`, true},
		{`{"metadata":{"unknown":1,"labels":{"a":"z"}}}`, true},
		{`{"spec":{"containers":[{"name":"c","unknown":1}]}}`, true},
		{`{"metadata":{"labels":{"a":"z"}},"metadata":{"labels":{"a":"y"}}}`, true},
		{`{"metadata":{"labels":"notamap"}}`, true},
		{`{"metadata":"notamap"}`, true},
		{`{"spec":{"containers":[{"name":"c","image":1}]}}`, true},
		{`{"status":{"conditions":[{"status":"False"}]}}`, true},
		{`{}`, true},
	}
	for _, tc := range cases {
		for _, directive := range []string{"", metav1.FieldValidationIgnore, metav1.FieldValidationWarn, metav1.FieldValidationStrict} {
			t.Run(fmt.Sprintf("%s/%s", tc.patch, directive), func(t *testing.T) {
				pruned := comparePatchOutcomes(t, testPod(), []byte(tc.patch), directive)
				if pruned != tc.wantPruned {
					t.Errorf("pruned = %v, want %v", pruned, tc.wantPruned)
				}
			})
		}
	}
}
