/*
Copyright 2014 The Kubernetes Authors.

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

package event

import (
	"context"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func TestGetAttrs(t *testing.T) {
	eventA := &api.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "f0118",
			Namespace: "default",
		},
		InvolvedObject: api.ObjectReference{
			Kind:            "Pod",
			Name:            "foo",
			Namespace:       "baz",
			UID:             "long uid string",
			APIVersion:      "v1",
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason: "ForTesting",
		Source: api.EventSource{Component: "test"},
		Type:   api.EventTypeNormal,
	}
	field := ToSelectableFields(eventA)
	expectA := fields.Set{
		"metadata.name":                  "f0118",
		"metadata.namespace":             "default",
		"involvedObject.kind":            "Pod",
		"involvedObject.name":            "foo",
		"involvedObject.namespace":       "baz",
		"involvedObject.uid":             "long uid string",
		"involvedObject.apiVersion":      "v1",
		"involvedObject.resourceVersion": "0",
		"involvedObject.fieldPath":       "",
		"reason":                         "ForTesting",
		"reportingComponent":             "",
		"source":                         "test",
		"type":                           api.EventTypeNormal,
	}
	if e, a := expectA, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", cmp.Diff(e, a))
	}

	eventB := &api.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "f0118",
			Namespace: "default",
		},
		InvolvedObject: api.ObjectReference{
			Kind:            "Pod",
			Name:            "foo",
			Namespace:       "baz",
			UID:             "long uid string",
			APIVersion:      "v1",
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason:              "ForTesting",
		ReportingController: "test",
		Type:                api.EventTypeNormal,
	}
	field = ToSelectableFields(eventB)
	expectB := fields.Set{
		"metadata.name":                  "f0118",
		"metadata.namespace":             "default",
		"involvedObject.kind":            "Pod",
		"involvedObject.name":            "foo",
		"involvedObject.namespace":       "baz",
		"involvedObject.uid":             "long uid string",
		"involvedObject.apiVersion":      "v1",
		"involvedObject.resourceVersion": "0",
		"involvedObject.fieldPath":       "",
		"reason":                         "ForTesting",
		"reportingComponent":             "test",
		"source":                         "test",
		"type":                           api.EventTypeNormal,
	}
	if e, a := expectB, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", cmp.Diff(e, a))
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
	fset := ToSelectableFields(&api.Event{})
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"v1",
		"Event",
		fset,
		nil,
	)
}

func TestValidateUpdate(t *testing.T) {
	makeEvent := func(name string) *api.Event {
		return &api.Event{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       "default",
				ResourceVersion: "123",
			},
			InvolvedObject: api.ObjectReference{
				Kind:            "Pod",
				Name:            "foo",
				Namespace:       "default",
				UID:             "long uid string",
				APIVersion:      "v1",
				ResourceVersion: "0",
				FieldPath:       "",
			},
			Reason: "ForTesting",
			Source: api.EventSource{Component: "test"},
			Type:   api.EventTypeNormal,
		}
	}
	eventA := makeEvent("eventA")
	eventB := makeEvent("eventB")
	errList := Strategy.ValidateUpdate(context.Background(), eventA, eventB)
	if len(errList) == 0 {
		t.Errorf("ValidateUpdate should fail on name change")
	}
}
