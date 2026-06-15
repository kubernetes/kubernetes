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

package meta

import (
	"context"
	"strings"
	"testing"
	"time"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
)

func RunObjectMetaTestCases[T runtime.Object](t *testing.T, ctx context.Context, baseObj T, strategy rest.RESTCreateStrategy, options ...apitesting.ValidationTestConfig) {
	t.Helper()
	fldPath := field.NewPath("metadata")
	trueVar := true

	// TODO: Remove MarkFromImperative from expected errors after adding corresponding declarative validation tags on ObjectMeta.
	testCases := []struct {
		Name         string
		Modify       func(metav1.Object)
		ExpectedErrs field.ErrorList
	}{
		{
			Name:         "valid: baseline",
			Modify:       func(meta metav1.Object) {},
			ExpectedErrs: field.ErrorList{},
		},
		{
			Name: "annotations: invalid key",
			Modify: func(meta metav1.Object) {
				meta.SetAnnotations(map[string]string{
					"invalid/key/format": "value",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("annotations"), "", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			Name: "annotations: too long",
			Modify: func(meta metav1.Object) {
				meta.SetAnnotations(map[string]string{
					"a": string(make([]byte, apivalidation.TotalAnnotationSizeLimitB)),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.TooLong(fldPath.Child("annotations"), "", apivalidation.TotalAnnotationSizeLimitB).MarkFromImperative(),
			},
		},
		{
			Name: "valid: annotations: at total size limit",
			Modify: func(meta metav1.Object) {
				meta.SetAnnotations(map[string]string{
					"a": string(make([]byte, apivalidation.TotalAnnotationSizeLimitB-1)),
				})
			},
			ExpectedErrs: field.ErrorList{},
		},
		{
			Name: "generation: negative",
			Modify: func(meta metav1.Object) {
				meta.SetGeneration(-1)
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("generation"), "", "").WithOrigin("minimum").MarkFromImperative(),
			},
		},
		{
			Name: "valid: generation: zero",
			Modify: func(meta metav1.Object) {
				meta.SetGeneration(0)
			},
			ExpectedErrs: field.ErrorList{},
		},
		{
			Name: "ownerReferences: empty apiVersion",
			Modify: func(meta metav1.Object) {
				meta.SetOwnerReferences([]metav1.OwnerReference{
					mkOwnerReference(tweakAPIVersion("")),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("ownerReferences").Child("apiVersion"), "", "version must not be empty").MarkFromImperative(),
			},
		},
		{
			Name: "ownerReferences: empty kind",
			Modify: func(meta metav1.Object) {
				meta.SetOwnerReferences([]metav1.OwnerReference{
					mkOwnerReference(tweakKind("")),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("ownerReferences").Child("kind"), "", "must not be empty").MarkFromImperative(),
			},
		},
		{
			Name: "ownerReferences: empty name",
			Modify: func(meta metav1.Object) {
				meta.SetOwnerReferences([]metav1.OwnerReference{
					mkOwnerReference(tweakName("")),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("ownerReferences").Child("name"), "", "must not be empty").MarkFromImperative(),
			},
		},
		{
			Name: "ownerReferences: empty uid",
			Modify: func(meta metav1.Object) {
				meta.SetOwnerReferences([]metav1.OwnerReference{
					mkOwnerReference(tweakUID("")),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("ownerReferences").Child("uid"), "", "must not be empty").MarkFromImperative(),
			},
		},
		{
			Name: "ownerReferences: event is disallowed",
			Modify: func(meta metav1.Object) {
				meta.SetOwnerReferences([]metav1.OwnerReference{
					mkOwnerReference(tweakKind("Event")),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("ownerReferences"), "", "v1, Kind=Event is disallowed from being an owner").MarkFromImperative(),
			},
		},
		{
			Name: "ownerReferences: multiple controllers",
			Modify: func(meta metav1.Object) {
				meta.SetOwnerReferences([]metav1.OwnerReference{
					mkOwnerReference(tweakName("name1"), tweakController(&trueVar)),
					mkOwnerReference(tweakName("name2"), tweakUID("uid-2"), tweakController(&trueVar)),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("ownerReferences"), "", "Only one reference can have Controller set to true. Found \"true\" in references for Pod/name1 and Pod/name2").MarkFromImperative(),
			},
		},
		{
			Name: "finalizers: conflicting",
			Modify: func(meta metav1.Object) {
				meta.SetFinalizers([]string{metav1.FinalizerOrphanDependents, metav1.FinalizerDeleteDependents})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("finalizers"), "", "finalizer orphan and foregroundDeletion cannot be both set").MarkFromImperative(),
			},
		},
		{
			Name: "finalizers: invalid format",
			Modify: func(meta metav1.Object) {
				meta.SetFinalizers([]string{"invalid/format/slash"})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("finalizers"), "", "").MarkFromImperative(),
			},
		},
		{
			Name: "finalizers: name too long",
			Modify: func(meta metav1.Object) {
				meta.SetFinalizers([]string{strings.Repeat("a", 317)})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("finalizers"), "", "").MarkFromImperative(),
			},
		},
		{
			Name: "managedFields: invalid operation",
			Modify: func(meta metav1.Object) {
				meta.SetManagedFields([]metav1.ManagedFieldsEntry{
					mkManagedFieldsEntry(tweakOperation("Invalid")),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("managedFields").Index(0).Child("operation"), "", "must be `Apply` or `Update`").MarkFromImperative(),
			},
		},
		{
			Name: "managedFields: invalid fieldsType",
			Modify: func(meta metav1.Object) {
				meta.SetManagedFields([]metav1.ManagedFieldsEntry{
					mkManagedFieldsEntry(tweakFieldsType("Invalid")),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("managedFields").Index(0).Child("fieldsType"), "", "must be `FieldsV1`").MarkFromImperative(),
			},
		},
		{
			Name: "managedFields: manager too long",
			Modify: func(meta metav1.Object) {
				meta.SetManagedFields([]metav1.ManagedFieldsEntry{
					mkManagedFieldsEntry(tweakManager(strings.Repeat("a", 129))),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.TooLong(fldPath.Child("managedFields").Index(0).Child("manager"), "", 128).MarkFromImperative(),
			},
		},
		{
			Name: "valid: managedFields: manager at max length",
			Modify: func(meta metav1.Object) {
				meta.SetManagedFields([]metav1.ManagedFieldsEntry{
					mkManagedFieldsEntry(tweakManager(strings.Repeat("a", 128))),
				})
			},
			ExpectedErrs: field.ErrorList{},
		},
		{
			Name: "managedFields: subresource too long",
			Modify: func(meta metav1.Object) {
				meta.SetManagedFields([]metav1.ManagedFieldsEntry{
					mkManagedFieldsEntry(tweakSubresource(strings.Repeat("a", 257))),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.TooLong(fldPath.Child("managedFields").Index(0).Child("subresource"), "", 0).MarkFromImperative(),
			},
		},
		{
			Name: "valid: managedFields: subresource at max length",
			Modify: func(meta metav1.Object) {
				meta.SetManagedFields([]metav1.ManagedFieldsEntry{
					mkManagedFieldsEntry(tweakSubresource(strings.Repeat("a", 256))),
				})
			},
			ExpectedErrs: field.ErrorList{},
		},
		{
			Name: "labels: invalid key format",
			Modify: func(meta metav1.Object) {
				meta.SetLabels(map[string]string{
					"a/b/c": "value",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("labels"), "", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			Name: "labels: key too long",
			Modify: func(meta metav1.Object) {
				meta.SetLabels(map[string]string{
					strings.Repeat("a", 317): "value",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("labels"), "", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			Name: "labels: invalid value format",
			Modify: func(meta metav1.Object) {
				meta.SetLabels(map[string]string{
					"key": "a!",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("labels"), "", "").WithOrigin("format=k8s-label-value").MarkFromImperative(),
			},
		},
		{
			Name: "labels: value too long",
			Modify: func(meta metav1.Object) {
				meta.SetLabels(map[string]string{
					"key": strings.Repeat("a", 64),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("labels"), "", "").WithOrigin("format=k8s-label-value").MarkFromImperative(),
			},
		},
		{
			Name: "valid: labels: value at max length",
			Modify: func(meta metav1.Object) {
				meta.SetLabels(map[string]string{
					"key": strings.Repeat("a", 63),
				})
			},
			ExpectedErrs: field.ErrorList{},
		},
	}

	if strategy.NamespaceScoped() {
		testCases = append(testCases, []struct {
			Name         string
			Modify       func(metav1.Object)
			ExpectedErrs field.ErrorList
		}{
			{
				Name: "namespace: empty",
				Modify: func(meta metav1.Object) {
					meta.SetNamespace("")
				},
				ExpectedErrs: field.ErrorList{
					field.Required(fldPath.Child("namespace"), "").MarkFromImperative(),
				},
			},
			{
				Name: "namespace: invalid dns label format",
				Modify: func(meta metav1.Object) {
					meta.SetNamespace("foo.bar")
				},
				ExpectedErrs: field.ErrorList{
					field.Invalid(fldPath.Child("namespace"), "", "").MarkFromImperative(),
				},
			},
			{
				Name: "namespace: too long",
				Modify: func(meta metav1.Object) {
					meta.SetNamespace(strings.Repeat("a", 64))
				},
				ExpectedErrs: field.ErrorList{
					field.Invalid(fldPath.Child("namespace"), "", "").MarkFromImperative(),
				},
			},
		}...)
	} else {
		testCases = append(testCases, struct {
			Name         string
			Modify       func(metav1.Object)
			ExpectedErrs field.ErrorList
		}{
			Name: "namespace: forbidden",
			Modify: func(meta metav1.Object) {
				meta.SetNamespace("default")
			},
			ExpectedErrs: field.ErrorList{
				field.Forbidden(fldPath.Child("namespace"), "not allowed on this type").MarkFromImperative(),
			},
		})
	}

	for _, tc := range testCases {
		t.Run("objectmeta: "+tc.Name, func(t *testing.T) {
			obj := baseObj.DeepCopyObject().(T)
			if accessor, err := apimeta.Accessor(obj); err == nil {
				tc.Modify(accessor)
			} else {
				t.Fatalf("failed to get accessor: %v", err)
			}
			apitesting.VerifyValidationEquivalence(t, ctx, obj, strategy, tc.ExpectedErrs, options...)
		})
	}
}

func RunObjectMetaUpdateTestCases[T runtime.Object](t *testing.T, ctx context.Context, baseObj T, strategy rest.RESTUpdateStrategy, options ...apitesting.ValidationTestConfig) {
	t.Helper()
	fldPath := field.NewPath("metadata")
	t1 := metav1.NewTime(time.Unix(1000, 0).UTC())
	t2 := metav1.NewTime(time.Unix(2000, 0).UTC())

	// TODO: Remove MarkFromImperative from expected errors after adding corresponding declarative validation tags on ObjectMeta.
	testCases := []struct {
		Name         string
		Modify       func(old, new metav1.Object)
		ExpectedErrs field.ErrorList
	}{
		{
			Name:         "update: valid: baseline",
			Modify:       func(old, new metav1.Object) {},
			ExpectedErrs: field.ErrorList{},
		},
		{
			Name: "update: annotations: invalid key",
			Modify: func(old, new metav1.Object) {
				new.SetAnnotations(map[string]string{
					"invalid/key/format": "value",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("annotations"), "", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			Name: "update: annotations: too long",
			Modify: func(old, new metav1.Object) {
				new.SetAnnotations(map[string]string{
					"a": string(make([]byte, apivalidation.TotalAnnotationSizeLimitB)),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.TooLong(fldPath.Child("annotations"), "", 0).MarkFromImperative(),
			},
		},
		{
			Name: "update: generation: decremented",
			Modify: func(old, new metav1.Object) {
				old.SetGeneration(5)
				new.SetGeneration(4)
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("generation"), "", "must not be decremented").MarkFromImperative(),
			},
		},
		{
			Name: "update: resourceVersion: missing",
			Modify: func(old, new metav1.Object) {
				new.SetResourceVersion("")
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("resourceVersion"), "", "must be specified for an update").MarkFromImperative(),
			},
		},
		{
			Name: "update: uid: immutable",
			Modify: func(old, new metav1.Object) {
				old.SetUID("uid-1")
				new.SetUID("uid-2")
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("uid"), "", "field is immutable").MarkFromImperative(),
			},
		},
		{
			Name: "update: creationTimestamp: immutable",
			Modify: func(old, new metav1.Object) {
				old.SetCreationTimestamp(t1)
				new.SetCreationTimestamp(t2)
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("creationTimestamp"), "", "field is immutable").MarkFromImperative(),
			},
		},
		{
			Name: "update: deletionTimestamp: immutable",
			Modify: func(old, new metav1.Object) {
				old.SetDeletionTimestamp(nil)
				new.SetDeletionTimestamp(&t2)
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("deletionTimestamp"), "", "field is immutable").MarkFromImperative(),
			},
		},
		{
			Name: "update: deletionGracePeriodSeconds: immutable",
			Modify: func(old, new metav1.Object) {
				g1 := int64(30)
				g2 := int64(40)
				old.SetDeletionGracePeriodSeconds(&g1)
				new.SetDeletionGracePeriodSeconds(&g2)
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("deletionGracePeriodSeconds"), "", "field is immutable").MarkFromImperative(),
			},
		},
		{
			Name: "update: finalizers: no new finalizers if deleted",
			Modify: func(old, new metav1.Object) {
				old.SetResourceVersion("1")
				new.SetResourceVersion("2")
				old.SetDeletionTimestamp(&t1)
				new.SetDeletionTimestamp(&t1)
				old.SetFinalizers([]string{"example.com/a"})
				new.SetFinalizers([]string{"example.com/a", "example.com/b"})
			},
			ExpectedErrs: field.ErrorList{
				field.Forbidden(fldPath.Child("finalizers"), "no new finalizers can be added if the object is being deleted, found new finalizers []string{\"example.com/b\"}").MarkFromImperative(),
			},
		},
		{
			Name: "update: finalizers: name too long",
			Modify: func(old, new metav1.Object) {
				new.SetFinalizers([]string{strings.Repeat("a", 317)})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("finalizers"), "", "").MarkFromImperative(),
			},
		},
		{
			Name: "update: managedFields: subresource too long",
			Modify: func(old, new metav1.Object) {
				new.SetManagedFields([]metav1.ManagedFieldsEntry{
					mkManagedFieldsEntry(tweakSubresource(strings.Repeat("a", 257))),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.TooLong(fldPath.Child("managedFields").Index(0).Child("subresource"), "", 0).MarkFromImperative(),
			},
		},
		{
			Name: "update: labels: invalid key format",
			Modify: func(old, new metav1.Object) {
				new.SetLabels(map[string]string{
					"a/b/c": "value",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("labels"), "", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			Name: "update: labels: key too long",
			Modify: func(old, new metav1.Object) {
				new.SetLabels(map[string]string{
					strings.Repeat("a", 317): "value",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("labels"), "", "").WithOrigin("format=k8s-label-key").MarkFromImperative(),
			},
		},
		{
			Name: "update: labels: invalid value format",
			Modify: func(old, new metav1.Object) {
				new.SetLabels(map[string]string{
					"key": "a!",
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("labels"), "", "").WithOrigin("format=k8s-label-value").MarkFromImperative(),
			},
		},
		{
			Name: "update: labels: value too long",
			Modify: func(old, new metav1.Object) {
				new.SetLabels(map[string]string{
					"key": strings.Repeat("a", 64),
				})
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("labels"), "", "").WithOrigin("format=k8s-label-value").MarkFromImperative(),
			},
		},
	}

	if strategy.NamespaceScoped() {
		testCases = append(testCases, struct {
			Name         string
			Modify       func(old, new metav1.Object)
			ExpectedErrs field.ErrorList
		}{
			Name: "update: namespace: immutable",
			Modify: func(old, new metav1.Object) {
				old.SetResourceVersion("1")
				new.SetResourceVersion("2")
				old.SetNamespace("ns-one")
				new.SetNamespace("ns-two")
			},
			ExpectedErrs: field.ErrorList{
				field.Invalid(fldPath.Child("namespace"), "", "field is immutable").MarkFromImperative(),
			},
		})
	}

	for _, tc := range testCases {
		t.Run("objectmeta: "+tc.Name, func(t *testing.T) {
			currOld := baseObj.DeepCopyObject().(T)
			currNew := baseObj.DeepCopyObject().(T)
			newAcc, _ := apimeta.Accessor(currNew)
			oldAcc, _ := apimeta.Accessor(currOld)
			oldAcc.SetResourceVersion("1")
			newAcc.SetResourceVersion("2")
			tc.Modify(oldAcc, newAcc)
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, currNew, currOld, strategy, tc.ExpectedErrs, options...)
		})
	}
}

func mkOwnerReference(tweaks ...func(*metav1.OwnerReference)) metav1.OwnerReference {
	ref := metav1.OwnerReference{
		APIVersion: "v1",
		Kind:       "Pod",
		Name:       "name",
		UID:        "uid-1",
	}
	for _, tweak := range tweaks {
		tweak(&ref)
	}
	return ref
}

func mkManagedFieldsEntry(tweaks ...func(*metav1.ManagedFieldsEntry)) metav1.ManagedFieldsEntry {
	entry := metav1.ManagedFieldsEntry{
		Operation:  "Update",
		FieldsType: "FieldsV1",
	}
	for _, tweak := range tweaks {
		tweak(&entry)
	}
	return entry
}

func tweakAPIVersion(v string) func(*metav1.OwnerReference) {
	return func(r *metav1.OwnerReference) {
		r.APIVersion = v
	}
}

func tweakKind(k string) func(*metav1.OwnerReference) {
	return func(r *metav1.OwnerReference) {
		r.Kind = k
	}
}

func tweakName(n string) func(*metav1.OwnerReference) {
	return func(r *metav1.OwnerReference) {
		r.Name = n
	}
}

func tweakUID(u types.UID) func(*metav1.OwnerReference) {
	return func(r *metav1.OwnerReference) {
		r.UID = u
	}
}

func tweakController(c *bool) func(*metav1.OwnerReference) {
	return func(r *metav1.OwnerReference) {
		r.Controller = c
	}
}

func tweakOperation(op metav1.ManagedFieldsOperationType) func(*metav1.ManagedFieldsEntry) {
	return func(m *metav1.ManagedFieldsEntry) {
		m.Operation = op
	}
}

func tweakFieldsType(ft string) func(*metav1.ManagedFieldsEntry) {
	return func(m *metav1.ManagedFieldsEntry) {
		m.FieldsType = ft
	}
}

func tweakManager(mng string) func(*metav1.ManagedFieldsEntry) {
	return func(m *metav1.ManagedFieldsEntry) {
		m.Manager = mng
	}
}

func tweakSubresource(sub string) func(*metav1.ManagedFieldsEntry) {
	return func(m *metav1.ManagedFieldsEntry) {
		m.Subresource = sub
	}
}
