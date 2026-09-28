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

package internal

import (
	"fmt"
	"reflect"
	"testing"

	"sigs.k8s.io/structured-merge-diff/v7/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v7/typed"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type recordingTypeConverter struct {
	TypeConverter
	sawManagedFields bool
	calls            int
}

func (r *recordingTypeConverter) ObjectToTyped(obj runtime.Object, opts ...typed.ValidationOptions) (*typed.TypedValue, error) {
	r.calls++
	if accessor, err := meta.Accessor(obj); err == nil && len(accessor.GetManagedFields()) > 0 {
		r.sawManagedFields = true
	}
	return r.TypeConverter.ObjectToTyped(obj, opts...)
}

type funcObjectConvertor struct {
	convertToVersion func(in runtime.Object, gv runtime.GroupVersioner) (runtime.Object, error)
}

func (f funcObjectConvertor) ConvertToVersion(in runtime.Object, gv runtime.GroupVersioner) (runtime.Object, error) {
	return f.convertToVersion(in, gv)
}

func (funcObjectConvertor) Convert(_, _, _ interface{}) error {
	return fmt.Errorf("not implemented")
}

func (funcObjectConvertor) ConvertFieldLabel(_ schema.GroupVersionKind, _, _ string) (string, string, error) {
	return "", "", fmt.Errorf("not implemented")
}

type noopDefaulter struct{}

func (noopDefaulter) Default(_ runtime.Object) {}

func TestStructuredMergeManagerUpdateSkipsLiveManagedFields(t *testing.T) {
	gv := schema.GroupVersion{Group: "apps", Version: "v1"}
	convertors := []struct {
		name string
		oc   runtime.ObjectConvertor
	}{
		{
			name: "copying",
			oc: funcObjectConvertor{convertToVersion: func(in runtime.Object, _ runtime.GroupVersioner) (runtime.Object, error) {
				return in.DeepCopyObject(), nil
			}},
		},
		{
			name: "identity",
			oc: funcObjectConvertor{convertToVersion: func(in runtime.Object, _ runtime.GroupVersioner) (runtime.Object, error) {
				return in, nil
			}},
		},
	}

	for _, backing := range benchBackings {
		for _, conv := range convertors {
			t.Run(fmt.Sprintf("%s/%s", backing.name, conv.name), func(t *testing.T) {
				tc := &recordingTypeConverter{TypeConverter: testTypeConverter}
				mgr, err := NewStructuredMergeManager(tc, conv.oc, noopDefaulter{}, gv, gv, nil)
				if err != nil {
					t.Fatalf("NewStructuredMergeManager: %v", err)
				}

				liveObj, managed := buildLiveWithManagedFields(t, backing.build("apps/v1", 10))
				liveAccessor, err := meta.Accessor(liveObj)
				if err != nil {
					t.Fatalf("meta.Accessor(liveObj): %v", err)
				}
				wantLiveManagedFields := append([]metav1.ManagedFieldsEntry(nil), liveAccessor.GetManagedFields()...)
				if len(wantLiveManagedFields) == 0 {
					t.Fatalf("expected liveObj to have non-empty managedFields")
				}

				newObj := backing.build("apps/v1", 11)
				if _, _, err := mgr.Update(liveObj, newObj, managed, "updater"); err != nil {
					t.Fatalf("Update: %v", err)
				}

				if tc.calls != 2 {
					t.Errorf("expected 2 ObjectToTyped calls, got %d", tc.calls)
				}
				if tc.sawManagedFields {
					t.Errorf("ObjectToTyped received an object with non-empty managedFields")
				}
				if got := liveAccessor.GetManagedFields(); !reflect.DeepEqual(got, wantLiveManagedFields) {
					t.Errorf("liveObj managedFields mutated:\ngot:  %#v\nwant: %#v", got, wantLiveManagedFields)
				}
			})
		}
	}
}

func buildLiveWithManagedFields(t testing.TB, obj runtime.Object) (runtime.Object, Managed) {
	t.Helper()
	tv, err := testTypeConverter.ObjectToTyped(obj, typed.AllowDuplicates)
	if err != nil {
		t.Fatalf("ObjectToTyped: %v", err)
	}
	set, err := tv.ToFieldSet()
	if err != nil {
		t.Fatalf("ToFieldSet: %v", err)
	}
	fieldsBytes, err := SetToFields(*set)
	if err != nil {
		t.Fatalf("SetToFields: %v", err)
	}
	now := metav1.Now()
	entries := []metav1.ManagedFieldsEntry{
		{
			Manager:    "manager-1",
			Operation:  metav1.ManagedFieldsOperationUpdate,
			APIVersion: "apps/v1",
			Time:       &now,
			FieldsType: "FieldsV1",
			FieldsV1:   &fieldsBytes,
		},
		{
			Manager:    "manager-2",
			Operation:  metav1.ManagedFieldsOperationApply,
			APIVersion: "apps/v1",
			Time:       &now,
			FieldsType: "FieldsV1",
			FieldsV1:   &fieldsBytes,
		},
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		t.Fatalf("meta.Accessor: %v", err)
	}
	accessor.SetManagedFields(entries)
	decoded, err := DecodeManagedFields(entries)
	if err != nil {
		t.Fatalf("DecodeManagedFields: %v", err)
	}
	return obj, decoded
}

func BenchmarkStructuredMergeManagerUpdate(b *testing.B) {
	gv := schema.GroupVersion{Group: "apps", Version: "v1"}
	oc := funcObjectConvertor{convertToVersion: func(in runtime.Object, _ runtime.GroupVersioner) (runtime.Object, error) {
		if d, ok := in.(*benchDeployment); ok {
			cp := *d
			return &cp, nil
		}
		return in.DeepCopyObject(), nil
	}}
	mgr, err := NewStructuredMergeManager(testTypeConverter, oc, noopDefaulter{}, gv, gv, nil)
	if err != nil {
		b.Fatalf("NewStructuredMergeManager: %v", err)
	}

	for _, n := range []int{10, 100} {
		liveObj, baseManaged := buildLiveWithManagedFields(b, structuredDeployment("apps/v1", n))
		newObj := structuredDeployment("apps/v1", n+1)
		b.Run(fmt.Sprintf("structured/fields=%d", n), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fieldsCopy := make(fieldpath.ManagedFields, len(baseManaged.Fields()))
				for k, v := range baseManaged.Fields() {
					fieldsCopy[k] = v
				}
				managed := NewManaged(fieldsCopy, baseManaged.Times())
				if _, _, err := mgr.Update(liveObj, newObj, managed, "updater"); err != nil {
					b.Fatalf("Update: %v", err)
				}
			}
		})
	}
}
