/*
Copyright 2022 The Kubernetes Authors.

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

package testing

import (
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

// NewFakeObjectCreater implements ObjectCreater, it can create empty
// objects (unstructured) of the given GVK.
func NewFakeObjectCreater() runtime.ObjectCreater {
	return &fakeObjectCreater{}
}

type fakeObjectCreater struct{}

func (f *fakeObjectCreater) New(gvk schema.GroupVersionKind) (runtime.Object, error) {
	u := unstructured.Unstructured{Object: map[string]interface{}{}}
	u.SetAPIVersion(gvk.GroupVersion().String())
	u.SetKind(gvk.Kind)
	return &u, nil
}

type fakeObjectConvertor struct {
	converter  merge.Converter
	apiVersion fieldpath.APIVersion
}

//nolint:staticcheck,ineffassign // SA4009 backwards compatibility
func (c *fakeObjectConvertor) Convert(in, out, context interface{}) error {
	if typedValue, ok := in.(*typed.TypedValue); ok {
		var err error
		out, err = c.converter.Convert(typedValue, c.apiVersion)
		return err
	}
	return nil
}

func (c *fakeObjectConvertor) ConvertToVersion(in runtime.Object, _ runtime.GroupVersioner) (runtime.Object, error) {
	return in, nil
}

func (c *fakeObjectConvertor) ConvertFieldLabel(_ schema.GroupVersionKind, _, _ string) (string, string, error) {
	return "", "", errors.New("not implemented")
}

type fakeObjectDefaulter struct{}

func (d *fakeObjectDefaulter) Default(in runtime.Object) {}

type sameVersionConverter struct{}

func (sameVersionConverter) Convert(object *typed.TypedValue, version fieldpath.APIVersion) (*typed.TypedValue, error) {
	return object, nil
}

func (sameVersionConverter) IsMissingVersionError(error) bool {
	return false
}

type TestFieldManagerImpl struct {
	fieldManager *internal.FieldManager
	apiVersion   string
	emptyObj     runtime.Object
	liveObj      runtime.Object
}

// APIVersion of the object that we're tracking.
func (f *TestFieldManagerImpl) APIVersion() string {
	return f.apiVersion
}

// Reset resets the state of the liveObject by resetting it to an empty object.
func (f *TestFieldManagerImpl) Reset() {
	f.liveObj = f.emptyObj.DeepCopyObject()
}

// Live returns a copy of the current liveObject.
func (f *TestFieldManagerImpl) Live() runtime.Object {
	return f.liveObj.DeepCopyObject()
}

// Apply applies the given object on top of the current liveObj, for the
// given manager and force flag.
func (f *TestFieldManagerImpl) Apply(obj runtime.Object, manager string, force bool) error {
	out, err := f.fieldManager.Apply(f.liveObj, obj, manager, force)
	if err == nil {
		f.liveObj = out
	}
	return err
}

// Update will updates the managed fields in the liveObj based on the
// changes performed by the update.
func (f *TestFieldManagerImpl) Update(obj runtime.Object, manager string) error {
	out, err := f.fieldManager.Update(f.liveObj, obj, manager)
	if err == nil {
		f.liveObj = out
	}
	return err
}

// ManagedFields returns the list of existing managed fields for the
// liveObj.
func (f *TestFieldManagerImpl) ManagedFields() []metav1.ManagedFieldsEntry {
	accessor, err := meta.Accessor(f.liveObj)
	if err != nil {
		panic(fmt.Errorf("couldn't get accessor: %v", err))
	}

	return accessor.GetManagedFields()
}

// NewTestFieldManager creates a new manager for the given GVK.
func NewTestFieldManagerImpl(typeConverter fieldmanager.TypeConverter, gvk schema.GroupVersionKind, subresource string, chainFieldManager func(internal.Manager) internal.Manager) *TestFieldManagerImpl {
	apiVersion := fieldpath.APIVersion(gvk.GroupVersion().String())
	objectConverter := &fakeObjectConvertor{sameVersionConverter{}, apiVersion}
	f, err := internal.NewStructuredMergeManager(
		typeConverter,
		objectConverter,
		&fakeObjectDefaulter{},
		gvk.GroupVersion(),
		gvk.GroupVersion(),
		nil,
	)
	if err != nil {
		panic(err)
	}
	live := &unstructured.Unstructured{}
	live.SetKind(gvk.Kind)
	live.SetAPIVersion(gvk.GroupVersion().String())
	// This is different from `internal.NewDefaultFieldManager` because:
	// 1. We don't want to create a `internal.FieldManager`
	// 2. We don't want to use the CapManager that is tested separately with
	// a smaller than the default cap.
	f = internal.NewLastAppliedUpdater(
		internal.NewLastAppliedManager(
			internal.NewProbabilisticSkipNonAppliedManager(
				internal.NewBuildManagerInfoManager(
					internal.NewManagedFieldsUpdater(
						internal.NewStripMetaManager(f),
					), gvk.GroupVersion(), subresource,
				), NewFakeObjectCreater(), gvk, internal.DefaultTrackOnCreateProbability,
			), typeConverter, objectConverter, gvk.GroupVersion(),
		),
	)
	if chainFieldManager != nil {
		f = chainFieldManager(f)
	}
	return &TestFieldManagerImpl{
		fieldManager: internal.NewFieldManager(f, subresource),
		apiVersion:   gvk.GroupVersion().String(),
		emptyObj:     live,
		liveObj:      live.DeepCopyObject(),
	}
}
