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

package validation

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateEvent(t *testing.T) {
	table := []struct {
		*core.Event
		valid bool
	}{
		{
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test1",
					Namespace: "foo",
				},
				InvolvedObject: core.ObjectReference{
					Namespace: "bar",
					Kind:      "Pod",
				},
			},
			false,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test2",
					Namespace: "aoeu-_-aoeu",
				},
				InvolvedObject: core.ObjectReference{
					Namespace: "aoeu-_-aoeu",
					Kind:      "Pod",
				},
			},
			false,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test3",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
			},
			true,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test4",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Namespace",
				},
			},
			true,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test5",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "apps/v1",
					Kind:       "NoKind",
					Namespace:  metav1.NamespaceDefault,
				},
			},
			true,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test6",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "batch/v1",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			false,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test7",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "batch/v1",
					Kind:       "Job",
					Namespace:  metav1.NamespaceDefault,
				},
			},
			true,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test8",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "other/v1beta1",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			false,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test9",
					Namespace: "foo",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "other/v1beta1",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			true,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test10",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "batch",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			false,
		}, {
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test11",
					Namespace: "foo",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "batch/v1",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			true,
		},
		{
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test12",
					Namespace: "foo",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "other/v1beta1",
					Kind:       "FooBar",
					Namespace:  "bar",
				},
			},
			false,
		},
		{
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test13",
					Namespace: "",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "other/v1beta1",
					Kind:       "FooBar",
					Namespace:  "bar",
				},
			},
			false,
		},
		{
			&core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test14",
					Namespace: "foo",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "other/v1beta1",
					Kind:       "FooBar",
					Namespace:  "",
				},
			},
			false,
		},
	}

	for _, item := range table {
		if e, a := item.valid, len(ValidateEvent(item.Event)) == 0; e != a {
			t.Errorf("%v: expected %v, got %v: %v", item.Event.Name, e, a, ValidateEvent(item.Event))
		}
	}
}

func TestValidateNewEvent(t *testing.T) {
	someTime := metav1.MicroTime{Time: time.Unix(1505828956, 0)}
	table := []struct {
		*core.Event
		valid bool
		msg   string
	}{
		{
			Event: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime: someTime,
			},
			valid: false,
			msg:   "Old Event with EventTime should trigger new validation and fail",
		},
		{
			Event: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: metav1.NamespaceSystem,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Because",
			},
			valid: true,
			msg:   "Valid new Event",
		},
		{
			Event: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: metav1.NamespaceSystem,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "my-contr@ller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Because",
			},
			valid: false,
			msg:   "not qualified reportingController",
		},
		{
			Event: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: metav1.NamespaceSystem,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
				Action:              "Do",
				Reason:              "Because",
			},
			valid: false,
			msg:   "too long reporting instance",
		},
		{
			Event: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: metav1.NamespaceSystem,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
			},
			valid: false,
			msg:   "missing reason",
		},
		{
			Event: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: metav1.NamespaceSystem,
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Reason:              "Because",
			},
			valid: false,
			msg:   "missing action",
		},
		{
			Event: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Reason:              "Because",
			},
			valid: false,
			msg:   "missing namespace",
		},
		{
			Event: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Because",
				Message: `zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz`,
			},
			valid: false,
			msg:   "too long message",
		},
	}

	for _, item := range table {
		if e, a := item.valid, len(ValidateEvent(item.Event)) == 0; e != a {
			t.Errorf("%v: expected %v, got %v: %v", item.msg, e, a, ValidateEvent(item.Event))
		}
	}
}
