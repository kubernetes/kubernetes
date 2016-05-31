/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package api_test

import (
	"reflect"
	"testing"

	"github.com/google/gofuzz"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/meta/metatypes"
)

var _ meta.Object = &api.ObjectMeta{}

// TestFillObjectMetaSystemFields validates that system populated fields are set on an object
func TestFillObjectMetaSystemFields(t *testing.T) {
	ctx := api.NewDefaultContext()
	resource := api.ObjectMeta{}
	api.FillObjectMetaSystemFields(ctx, &resource)
	if resource.CreationTimestamp.Time.IsZero() {
		t.Errorf("resource.CreationTimestamp is zero")
	} else if len(resource.UID) == 0 {
		t.Errorf("resource.UID missing")
	}
}

// TestHasObjectMetaSystemFieldValues validates that true is returned if and only if all fields are populated
func TestHasObjectMetaSystemFieldValues(t *testing.T) {
	ctx := api.NewDefaultContext()
	resource := api.ObjectMeta{}
	if api.HasObjectMetaSystemFieldValues(&resource) {
		t.Errorf("the resource does not have all fields yet populated, but incorrectly reports it does")
	}
	api.FillObjectMetaSystemFields(ctx, &resource)
	if !api.HasObjectMetaSystemFieldValues(&resource) {
		t.Errorf("the resource does have all fields populated, but incorrectly reports it does not")
	}
}

func getObjectMetaAndOwnerReferences() (objectMeta api.ObjectMeta, metaOwnerReferences []metatypes.OwnerReference) {
	fuzz.New().NilChance(.5).NumElements(1, 5).Fuzz(&objectMeta)
	references := objectMeta.OwnerReferences
	metaOwnerReferences = make([]metatypes.OwnerReference, 0)
	for i := 0; i < len(references); i++ {
		metaOwnerReferences = append(metaOwnerReferences, metatypes.OwnerReference{
			Kind:       references[i].Kind,
			Name:       references[i].Name,
			UID:        references[i].UID,
			APIVersion: references[i].APIVersion,
			Controller: references[i].Controller,
		})
	}
	if len(references) == 0 {
		objectMeta.OwnerReferences = make([]api.OwnerReference, 0)
	}
	return objectMeta, metaOwnerReferences
}

func testGetOwnerReferences(t *testing.T) {
	meta, expected := getObjectMetaAndOwnerReferences()
	refs := meta.GetOwnerReferences()
	if !reflect.DeepEqual(refs, expected) {
		t.Errorf("expect %v\n got %v", expected, refs)
	}
}

func testSetOwnerReferences(t *testing.T) {
	expected, newRefs := getObjectMetaAndOwnerReferences()
	objectMeta := &api.ObjectMeta{}
	objectMeta.SetOwnerReferences(newRefs)
	if !reflect.DeepEqual(expected.OwnerReferences, objectMeta.OwnerReferences) {
		t.Errorf("expect: %#v\n got: %#v", expected.OwnerReferences, objectMeta.OwnerReferences)
	}
}

func TestAccessOwnerReferences(t *testing.T) {
	fuzzIter := 5
	for i := 0; i < fuzzIter; i++ {
		testGetOwnerReferences(t)
		testSetOwnerReferences(t)
	}
}
