/*
Copyright 2017 The Kubernetes Authors.

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

package v1beta1

import (
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

var scheme = runtime.NewScheme()

func init() {
	addKnownTypes(scheme)
	internalGV := schema.GroupVersion{Group: auditinternal.GroupName, Version: runtime.APIVersionInternal}
	scheme.AddKnownTypes(internalGV,
		&auditinternal.Event{},
	)
	RegisterConversions(scheme)
}

func TestConversionEventToInternal(t *testing.T) {
	time1 := time.Now()
	time2 := time.Now()
	testcases := []struct {
		desc     string
		old      *Event
		expected *auditinternal.Event
	}{
		{
			"StageTimestamp is empty",
			&Event{
				ObjectMeta: metav1.ObjectMeta{
					CreationTimestamp: metav1.NewTime(time1),
				},
			},
			&auditinternal.Event{
				StageTimestamp: metav1.NewMicroTime(time1),
			},
		},
		{
			"StageTimestamp is not empty",
			&Event{
				ObjectMeta: metav1.ObjectMeta{
					CreationTimestamp: metav1.NewTime(time1),
				},
				StageTimestamp: metav1.NewMicroTime(time2),
			},
			&auditinternal.Event{
				StageTimestamp: metav1.NewMicroTime(time2),
			},
		},
		{
			"RequestReceivedTimestamp is empty",
			&Event{
				Timestamp: metav1.NewTime(time1),
			},
			&auditinternal.Event{
				RequestReceivedTimestamp: metav1.NewMicroTime(time1),
			},
		},
		{
			"RequestReceivedTimestamp is not empty",
			&Event{
				Timestamp:                metav1.NewTime(time1),
				RequestReceivedTimestamp: metav1.NewMicroTime(time2),
			},
			&auditinternal.Event{
				RequestReceivedTimestamp: metav1.NewMicroTime(time2),
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			internal := &auditinternal.Event{}
			if err := scheme.Convert(tc.old, internal, nil); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(internal, tc.expected) {
				t.Errorf("expected\n\t%#v, got \n\t%#v", tc.expected, internal)
			}
		})
	}
}

func TestConversionInternalToEvent(t *testing.T) {
	now := time.Now()
	testcases := []struct {
		desc     string
		old      *auditinternal.Event
		expected *Event
	}{
		{
			"convert stageTimestamp",
			&auditinternal.Event{
				StageTimestamp: metav1.NewMicroTime(now),
			},
			&Event{
				ObjectMeta: metav1.ObjectMeta{
					CreationTimestamp: metav1.NewTime(now),
				},
				StageTimestamp: metav1.NewMicroTime(now),
			},
		},
		{
			"convert RequestReceivedTimestamp",
			&auditinternal.Event{
				RequestReceivedTimestamp: metav1.NewMicroTime(now),
			},
			&Event{
				Timestamp:                metav1.NewTime(now),
				RequestReceivedTimestamp: metav1.NewMicroTime(now),
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.desc, func(t *testing.T) {
			event := &Event{}
			if err := scheme.Convert(tc.old, event, nil); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(event, tc.expected) {
				t.Errorf("expected\n\t%#v, got \n\t%#v", tc.expected, event)
			}
		})
	}
}
