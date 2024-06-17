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

package events

import (
	"bytes"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"testing"
	"time"
)

func TestPrintObj(t *testing.T) {
	tests := []struct {
		printer  EventPrinter
		obj      runtime.Object
		expected string
	}{
		{
			printer: EventPrinter{
				NoHeaders:     false,
				AllNamespaces: false,
			},
			obj: &corev1.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar-000",
					Namespace: "foo",
				},
				InvolvedObject: corev1.ObjectReference{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "bar",
					Namespace:  "foo",
					UID:        "00000000-0000-0000-0000-000000000001",
				},
				Type:                corev1.EventTypeNormal,
				Reason:              "ScalingReplicaSet",
				Message:             "Scaled up replica set bar-002 to 1",
				ReportingController: "deployment-controller",
				EventTime:           metav1.NewMicroTime(time.Now().Add(-20 * time.Minute)),
				Series: &corev1.EventSeries{
					Count:            3,
					LastObservedTime: metav1.NewMicroTime(time.Now().Add(-12 * time.Minute)),
				},
			},
			expected: `LAST SEEN	TYPE	REASON	OBJECT	MESSAGE
12m (x3 over 20m)	Normal	ScalingReplicaSet	Deployment/bar	Scaled up replica set bar-002 to 1
`,
		},
		{
			printer: EventPrinter{
				NoHeaders:     false,
				AllNamespaces: true,
			},
			obj: &corev1.EventList{
				Items: []corev1.Event{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "bar-000",
							Namespace: "foo",
						},
						InvolvedObject: corev1.ObjectReference{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "bar",
							Namespace:  "foo",
							UID:        "00000000-0000-0000-0000-000000000001",
						},
						Type:                corev1.EventTypeNormal,
						Reason:              "ScalingReplicaSet",
						Message:             "Scaled up replica set bar-002 to 1",
						ReportingController: "deployment-controller",
						EventTime:           metav1.NewMicroTime(time.Now().Add(-20 * time.Minute)),
						Series: &corev1.EventSeries{
							Count:            3,
							LastObservedTime: metav1.NewMicroTime(time.Now().Add(-12 * time.Minute)),
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "bar-001",
							Namespace: "bar",
						},
						InvolvedObject: corev1.ObjectReference{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "bar2",
							Namespace:  "foo2",
							UID:        "00000000-0000-0000-0000-000000000001",
						},
						Type:                corev1.EventTypeNormal,
						Reason:              "ScalingReplicaSet",
						Message:             "Scaled up replica set bar-002 to 1",
						ReportingController: "deployment-controller",
						EventTime:           metav1.NewMicroTime(time.Now().Add(-15 * time.Minute)),
						Series: &corev1.EventSeries{
							Count:            3,
							LastObservedTime: metav1.NewMicroTime(time.Now().Add(-11 * time.Minute)),
						},
					},
				},
			},
			expected: `NAMESPACE	LAST SEEN	TYPE	REASON	OBJECT	MESSAGE
foo	12m (x3 over 20m)	Normal	ScalingReplicaSet	Deployment/bar	Scaled up replica set bar-002 to 1
bar	11m (x3 over 15m)	Normal	ScalingReplicaSet	Deployment/bar2	Scaled up replica set bar-002 to 1
`,
		},
		{
			printer: EventPrinter{
				NoHeaders:     true,
				AllNamespaces: false,
			},
			obj: &corev1.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar-000",
					Namespace: "foo",
				},
				InvolvedObject: corev1.ObjectReference{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "bar",
					Namespace:  "foo",
					UID:        "00000000-0000-0000-0000-000000000001",
				},
				Type:                corev1.EventTypeNormal,
				Reason:              "ScalingReplicaSet",
				Message:             "Scaled up replica set bar-002 to 1",
				ReportingController: "deployment-controller",
				EventTime:           metav1.NewMicroTime(time.Now().Add(-20 * time.Minute)),
				Series: &corev1.EventSeries{
					Count:            3,
					LastObservedTime: metav1.NewMicroTime(time.Now().Add(-12 * time.Minute)),
				},
			},
			expected: "12m (x3 over 20m)	Normal	ScalingReplicaSet	Deployment/bar	Scaled up replica set bar-002 to 1\n",
		},
		{
			printer: EventPrinter{
				NoHeaders:     false,
				AllNamespaces: true,
			},
			obj: &corev1.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar-000",
					Namespace: "foo",
				},
				InvolvedObject: corev1.ObjectReference{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "bar",
					Namespace:  "foo",
					UID:        "00000000-0000-0000-0000-000000000001",
				},
				Type:                corev1.EventTypeNormal,
				Reason:              "ScalingReplicaSet",
				Message:             "Scaled up replica set bar-002 to 1",
				ReportingController: "deployment-controller",
				EventTime:           metav1.NewMicroTime(time.Now().Add(-20 * time.Minute)),
				Series: &corev1.EventSeries{
					Count:            3,
					LastObservedTime: metav1.NewMicroTime(time.Now().Add(-12 * time.Minute)),
				},
			},
			expected: `NAMESPACE	LAST SEEN	TYPE	REASON	OBJECT	MESSAGE
foo	12m (x3 over 20m)	Normal	ScalingReplicaSet	Deployment/bar	Scaled up replica set bar-002 to 1
`,
		},
		{
			printer: EventPrinter{
				NoHeaders:     true,
				AllNamespaces: true,
			},
			obj: &corev1.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "bar-000",
					Namespace: "foo",
				},
				InvolvedObject: corev1.ObjectReference{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "bar",
					Namespace:  "foo",
					UID:        "00000000-0000-0000-0000-000000000001",
				},
				Type:                corev1.EventTypeNormal,
				Reason:              "ScalingReplicaSet",
				Message:             "Scaled up replica set bar-002 to 1",
				ReportingController: "deployment-controller",
				EventTime:           metav1.NewMicroTime(time.Now().Add(-20 * time.Minute)),
				Series: &corev1.EventSeries{
					Count:            3,
					LastObservedTime: metav1.NewMicroTime(time.Now().Add(-12 * time.Minute)),
				},
			},
			expected: `foo	12m (x3 over 20m)	Normal	ScalingReplicaSet	Deployment/bar	Scaled up replica set bar-002 to 1
`,
		},
		{
			printer: EventPrinter{
				NoHeaders:     false,
				AllNamespaces: false,
			},
			obj: &corev1.EventList{
				Items: []corev1.Event{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "bar-000",
							Namespace: "foo",
						},
						InvolvedObject: corev1.ObjectReference{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "bar\x1b",
							Namespace:  "foo",
						},
						Type:                "test\x1b",
						Reason:              "test\x1b",
						Message:             "\x1b",
						ReportingController: "deployment-controller",
						EventTime:           metav1.NewMicroTime(time.Now().Add(-20 * time.Minute)),
						Series: &corev1.EventSeries{
							Count:            3,
							LastObservedTime: metav1.NewMicroTime(time.Now().Add(-1 * time.Minute)),
						},
					},
				},
			},
			expected: `LAST SEEN	TYPE	REASON	OBJECT	MESSAGE
60s (x3 over 20m)	test^[	test^[	Deployment/bar^[	^[
`,
		},
	}

	for _, test := range tests {
		t.Run("", func(t *testing.T) {
			buffer := &bytes.Buffer{}
			if err := test.printer.PrintObj(test.obj, buffer); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if buffer.String() != test.expected {
				t.Errorf("\nexpected:\n'%s'\nsaw\n'%s'\n", test.expected, buffer.String())
			}
		})
	}
}
