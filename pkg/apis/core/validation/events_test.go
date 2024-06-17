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

	v1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	eventsv1beta1 "k8s.io/api/events/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestValidateEventForCoreV1Events(t *testing.T) {
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
		createErrs := ValidateEventCreate(item.Event, v1.SchemeGroupVersion)
		if e, a := item.valid, len(createErrs) == 0; e != a {
			t.Errorf("%v: expected %v, got %v: %v", item.Event.Name, e, a, createErrs)
		}
		updateErrs := ValidateEventUpdate(item.Event, &core.Event{}, v1.SchemeGroupVersion)
		if e, a := item.valid, len(updateErrs) == 0; e != a {
			t.Errorf("%v: expected %v, got %v: %v", item.Event.Name, e, a, updateErrs)
		}
	}
}

func TestValidateEventForNewV1beta1Events(t *testing.T) {
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
		createErrs := ValidateEventCreate(item.Event, eventsv1beta1.SchemeGroupVersion)
		if e, a := item.valid, len(createErrs) == 0; e != a {
			t.Errorf("%v: expected %v, got %v: %v", item.msg, e, a, createErrs)
		}
		updateErrs := ValidateEventUpdate(item.Event, &core.Event{}, eventsv1beta1.SchemeGroupVersion)
		if e, a := item.valid, len(updateErrs) == 0; e != a {
			t.Errorf("%v: expected %v, got %v: %v", item.msg, e, a, updateErrs)
		}
	}
}

func TestValidateEventCreateForNewV1Events(t *testing.T) {
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
				Type:                "Normal",
			},
			valid: true,
			msg:   "valid new event",
		},
		{
			Event: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
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
			msg:   "missing name in objectMeta",
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
			msg:   "missing namespace in objectMeta",
		},
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
			},
			valid: false,
			msg:   "missing EventTime",
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
				Type:                "invalid-type",
			},
			valid: false,
			msg:   "invalid type",
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
				Type:                "Normal",
				FirstTimestamp:      metav1.Time{Time: time.Unix(1505828956, 0)},
			},
			valid: false,
			msg:   "non-empty firstTimestamp",
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
				Type:                "Normal",
				LastTimestamp:       metav1.Time{Time: time.Unix(1505828956, 0)},
			},
			valid: false,
			msg:   "non-empty lastTimestamp",
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
				Type:                "Normal",
				Count:               123,
			},
			valid: false,
			msg:   "non-empty count",
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
				Type:                "Normal",
				Source: core.EventSource{
					Host: "host",
				},
			},
			valid: false,
			msg:   "non-empty source",
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
				Type:                "Normal",
				Series: &core.EventSeries{
					Count:            0,
					LastObservedTime: someTime,
				},
			},
			valid: false,
			msg:   "non-nil series with cound < 2",
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
				Type:                "Normal",
				Series: &core.EventSeries{
					Count: 2,
				},
			},
			valid: false,
			msg:   "non-nil series with empty lastObservedTime",
		},
	}

	for _, item := range table {
		createErrs := ValidateEventCreate(item.Event, eventsv1.SchemeGroupVersion)
		if e, a := item.valid, len(createErrs) == 0; e != a {
			t.Errorf("%v: expected %v, got %v: %v", item.msg, e, a, createErrs)
		}
	}
}

func TestValidateEventUpdateForNewV1Events(t *testing.T) {
	someTime := metav1.MicroTime{Time: time.Unix(1505828956, 0)}
	table := []struct {
		newEvent *core.Event
		oldEvent *core.Event
		valid    bool
		msg      string
	}{
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "2",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v2",
					Kind:       "Node",
				},
				Series: &core.EventSeries{
					Count:            2,
					LastObservedTime: someTime,
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "2",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v2",
					Kind:       "Node",
				},
				Series: &core.EventSeries{
					Count:            1,
					LastObservedTime: someTime,
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: true,
			msg:   "valid new updated event",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v2",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: false,
			msg:   "forbidden updates to involvedObject",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees-new",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: false,
			msg:   "forbidden updates to reason",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
				Message:             "new-message",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
				Message:             "message",
			},
			valid: false,
			msg:   "forbidden updates to message",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				Source: core.EventSource{
					Host: "host",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: false,
			msg:   "forbidden updates to source",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
				FirstTimestamp:      metav1.Time{Time: time.Unix(1505828956, 0)},
			},
			valid: false,
			msg:   "forbidden updates to firstTimestamp",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
				LastTimestamp:       metav1.Time{Time: time.Unix(1505828956, 0)},
			},
			valid: false,
			msg:   "forbidden updates to lastTimestamp",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
				Count:               2,
			},
			valid: false,
			msg:   "forbidden updates to count",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Warning",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: false,
			msg:   "forbidden updates to type",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           metav1.MicroTime{Time: time.Unix(1505828999, 0)},
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: false,
			msg:   "forbidden updates to eventTime",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Undo",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: false,
			msg:   "forbidden updates to action",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				Related: &core.ObjectReference{
					APIVersion: "v1",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: false,
			msg:   "forbidden updates to related",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller/new",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: false,
			msg:   "forbidden updates to reportingController",
		},
		{
			newEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz-new",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			oldEvent: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test",
					Namespace:       metav1.NamespaceSystem,
					ResourceVersion: "1",
				},
				InvolvedObject: core.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
				EventTime:           someTime,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			},
			valid: false,
			msg:   "forbidden updates to reportingInstance",
		},
	}

	for _, item := range table {
		updateErrs := ValidateEventUpdate(item.newEvent, item.oldEvent, eventsv1.SchemeGroupVersion)
		if e, a := item.valid, len(updateErrs) == 0; e != a {
			t.Errorf("%v: expected %v, got %v: %v", item.msg, e, a, updateErrs)
		}
	}
}

func TestEventV1EventTimeImmutability(t *testing.T) {
	testcases := []struct {
		Name  string
		Old   metav1.MicroTime
		New   metav1.MicroTime
		Valid bool
	}{
		{
			Name:  "noop microsecond precision",
			Old:   metav1.NewMicroTime(time.Unix(100, int64(5*time.Microsecond))),
			New:   metav1.NewMicroTime(time.Unix(100, int64(5*time.Microsecond))),
			Valid: true,
		},
		{
			Name:  "noop nanosecond precision",
			Old:   metav1.NewMicroTime(time.Unix(100, int64(5*time.Nanosecond))),
			New:   metav1.NewMicroTime(time.Unix(100, int64(5*time.Nanosecond))),
			Valid: true,
		},
		{
			Name:  "modify nanoseconds within the same microsecond",
			Old:   metav1.NewMicroTime(time.Unix(100, int64(5*time.Nanosecond))),
			New:   metav1.NewMicroTime(time.Unix(100, int64(6*time.Nanosecond))),
			Valid: true,
		},
		{
			Name:  "modify microseconds",
			Old:   metav1.NewMicroTime(time.Unix(100, int64(5*time.Microsecond))),
			New:   metav1.NewMicroTime(time.Unix(100, int64(5*time.Microsecond-time.Nanosecond))),
			Valid: false,
		},
		{
			Name:  "modify seconds",
			Old:   metav1.NewMicroTime(time.Unix(100, 0)),
			New:   metav1.NewMicroTime(time.Unix(101, 0)),
			Valid: false,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			oldEvent := &core.Event{
				ObjectMeta:          metav1.ObjectMeta{Name: "test", Namespace: metav1.NamespaceSystem, ResourceVersion: "2"},
				InvolvedObject:      core.ObjectReference{APIVersion: "v2", Kind: "Node"},
				Series:              &core.EventSeries{Count: 2, LastObservedTime: tc.Old},
				EventTime:           tc.Old,
				ReportingController: "k8s.io/my-controller",
				ReportingInstance:   "node-xyz",
				Action:              "Do",
				Reason:              "Yeees",
				Type:                "Normal",
			}

			newEvent := oldEvent.DeepCopy()
			newEvent.EventTime = tc.New

			updateErrs := ValidateEventUpdate(newEvent, oldEvent, eventsv1.SchemeGroupVersion)
			if e, a := tc.Valid, len(updateErrs) == 0; e != a {
				t.Errorf("%v: expected valid=%v, got %v: %v", tc.Valid, e, a, updateErrs)
			}
		})
	}
}
