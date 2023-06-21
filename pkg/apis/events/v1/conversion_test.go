/*
Copyright 2020 The Kubernetes Authors.

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

package v1_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/events/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/core"
	eventsv1 "k8s.io/kubernetes/pkg/apis/events/v1"
)

func Test_convert_v1_Event_To_core_Event(t *testing.T) {
	type args struct {
		in  *v1.Event
		out *core.Event
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		wantOut *core.Event
	}{
		{
			name: "Test_convert_v1_Event_To_core_Event",
			args: args{
				in: &v1.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "foo",
						Namespace: metav1.NamespaceDefault,
					},
					Regarding: corev1.ObjectReference{
						Kind:      "Pod",
						Namespace: metav1.NamespaceDefault,
						Name:      "foo",
					},
					DeprecatedSource: corev1.EventSource{
						Component: "controller-manager",
						Host:      "master1",
					},
					Note: "convert v1 event to core event",
				},
				out: &core.Event{},
			},
			wantErr: false,
			wantOut: &core.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: core.ObjectReference{
					Namespace: metav1.NamespaceDefault,
					Name:      "foo",
					Kind:      "Pod",
				},
				Source: core.EventSource{
					Component: "controller-manager",
					Host:      "master1",
				},
				Message: "convert v1 event to core event",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := eventsv1.Convert_v1_Event_To_core_Event(tt.args.in, tt.args.out, nil); (err != nil) != tt.wantErr {
				t.Errorf("Convert_v1_Event_To_core_Event() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.args.out, tt.wantOut); diff != "" {
				t.Errorf("Convert_v1_Event_To_core_Event() mismatch (-want +got):\n %s", diff)
			}
		})
	}
}

func Test_convert_core_Event_To_v1_Event(t *testing.T) {

	type args struct {
		in  *core.Event
		out *v1.Event
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		wantOut *v1.Event
	}{
		{
			name: "Test_convert_core_Event_To_v1_Event",
			args: args{
				in: &core.Event{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "foo",
						Namespace: metav1.NamespaceDefault,
					},
					InvolvedObject: core.ObjectReference{
						Namespace: metav1.NamespaceDefault,
						Name:      "foo",
						Kind:      "Pod",
					},
					Source: core.EventSource{
						Component: "controller-manager",
						Host:      "master1",
					},
					Message: "convert core event to v1 event",
				},
				out: &v1.Event{},
			},
			wantErr: false,
			wantOut: &v1.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: metav1.NamespaceDefault,
				},
				Regarding: corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: metav1.NamespaceDefault,
					Name:      "foo",
				},
				DeprecatedSource: corev1.EventSource{
					Component: "controller-manager",
					Host:      "master1",
				},
				Note: "convert core event to v1 event",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := eventsv1.Convert_core_Event_To_v1_Event(tt.args.in, tt.args.out, nil); (err != nil) != tt.wantErr {
				t.Errorf("Convert_core_Event_To_v1_Event() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.args.out, tt.wantOut); diff != "" {
				t.Errorf("Convert_core_Event_To_v1_Event() mismatch (-want +got):\n %s", diff)
			}
		})
	}
}
