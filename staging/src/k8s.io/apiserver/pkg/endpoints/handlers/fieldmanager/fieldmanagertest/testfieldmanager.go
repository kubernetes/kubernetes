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

package fieldmanagertest

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

var builtinConverter = func() fieldmanager.TypeConverter {
	data, err := ioutil.ReadFile(filepath.Join(
		strings.Repeat(".."+string(filepath.Separator), 8),
		"api", "openapi-spec", "swagger.json"))
	if err != nil {
		panic(err)
	}
	spec := spec.Swagger{}
	if err := json.Unmarshal(data, &spec); err != nil {
		panic(err)
	}
	tc, err := fieldmanager.NewTypeConverter(&spec, false)
	if err != nil {
		panic(fmt.Errorf("Failed to build TypeConverter: %v", err))
	}
	return tc
}()

// NewBuiltinTypeConverter creates a TypeConverter with all the built-in
// types defined, given the committed kubernetes swagger.json.
func NewBuiltinTypeConverter() fieldmanager.TypeConverter {
	return builtinConverter
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

// TestFieldManager is a FieldManager that can be used in test to
// simulate the behavior of Server-Side Apply and field tracking. This
// also has a few methods to get a sense of the state of the object.
//
// This TestFieldManager uses a series of "fake" objects to simulate
// some behavior which come with the limitation that you can only use
// one version since there is no version conversion logic.
//
// You can use this rather than NewDefaultTestFieldManager if you want
// to specify either a sub-resource, or a set of modified Manager to
// test them specifically.
type TestFieldManager struct {
	fieldManager *fieldmanager.FieldManager
	apiVersion   string
	emptyObj     runtime.Object
	liveObj      runtime.Object
}

// NewDefaultTestFieldManager returns a new TestFieldManager built for
// the given gvk, on the main resource.
func NewDefaultTestFieldManager(gvk schema.GroupVersionKind) TestFieldManager {
	return NewTestFieldManager(gvk, "", nil)
}

// NewTestFieldManager creates a new manager for the given GVK.
func NewTestFieldManager(gvk schema.GroupVersionKind, subresource string, chainFieldManager func(fieldmanager.Manager) fieldmanager.Manager) TestFieldManager {
	typeConverter := NewBuiltinTypeConverter()
	apiVersion := fieldpath.APIVersion(gvk.GroupVersion().String())
	objectConverter := &fakeObjectConvertor{sameVersionConverter{}, apiVersion}
	f, err := fieldmanager.NewStructuredMergeManager(
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
	f = fieldmanager.NewLastAppliedUpdater(
		fieldmanager.NewLastAppliedManager(
			fieldmanager.NewProbabilisticSkipNonAppliedManager(
				fieldmanager.NewBuildManagerInfoManager(
					fieldmanager.NewManagedFieldsUpdater(
						fieldmanager.NewStripMetaManager(f),
					), gvk.GroupVersion(), subresource,
				), NewFakeObjectCreater(), gvk, fieldmanager.DefaultTrackOnCreateProbability,
			), typeConverter, objectConverter, gvk.GroupVersion(),
		),
	)
	if chainFieldManager != nil {
		f = chainFieldManager(f)
	}
	return TestFieldManager{
		fieldManager: fieldmanager.NewFieldManager(f, subresource),
		apiVersion:   gvk.GroupVersion().String(),
		emptyObj:     live,
		liveObj:      live.DeepCopyObject(),
	}
}

// APIVersion of the object that we're tracking.
func (f *TestFieldManager) APIVersion() string {
	return f.apiVersion
}

// Reset resets the state of the liveObject by resetting it to an empty object.
func (f *TestFieldManager) Reset() {
	f.liveObj = f.emptyObj.DeepCopyObject()
}

// Live returns a copy of the current liveObject.
func (f *TestFieldManager) Live() runtime.Object {
	return f.liveObj.DeepCopyObject()
}

// Apply applies the given object on top of the current liveObj, for the
// given manager and force flag.
func (f *TestFieldManager) Apply(obj runtime.Object, manager string, force bool) error {
	out, err := f.fieldManager.Apply(f.liveObj, obj, manager, force)
	if err == nil {
		f.liveObj = out
	}
	return err
}

// Update will updates the managed fields in the liveObj based on the
// changes performed by the update.
func (f *TestFieldManager) Update(obj runtime.Object, manager string) error {
	out, err := f.fieldManager.Update(f.liveObj, obj, manager)
	if err == nil {
		f.liveObj = out
	}
	return err
}

// ManagedFields returns the list of existing managed fields for the
// liveObj.
func (f *TestFieldManager) ManagedFields() []metav1.ManagedFieldsEntry {
	accessor, err := meta.Accessor(f.liveObj)
	if err != nil {
		panic(fmt.Errorf("couldn't get accessor: %v", err))
	}

	return accessor.GetManagedFields()
}
