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

package events

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestEventGetAttrs(t *testing.T) {
	eventA := &core.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "f0118",
			Namespace: "default",
		},
		InvolvedObject: core.ObjectReference{
			Kind:            "Pod",
			Name:            "foo",
			Namespace:       "baz",
			UID:             "long uid string",
			APIVersion:      "v1",
			ResourceVersion: "0",
			FieldPath:       "",
		},
		Reason: "ForTesting",
		Source: core.EventSource{Component: "test"},
		Type:   core.EventTypeNormal,
	}
	field := EventToSelectableFields(eventA)
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
		"type":                           core.EventTypeNormal,
	}
	if e, a := expectA, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", cmp.Diff(e, a))
	}

	eventB := &core.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "f0118",
			Namespace: "default",
		},
		InvolvedObject: core.ObjectReference{
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
		Type:                core.EventTypeNormal,
	}
	field = EventToSelectableFields(eventB)
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
		"type":                           core.EventTypeNormal,
	}
	if e, a := expectB, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", cmp.Diff(e, a))
	}
}
