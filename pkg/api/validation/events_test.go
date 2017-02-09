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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
)

func TestValidateEvent(t *testing.T) {
	table := []struct {
		*api.Event
		valid bool
	}{
		{
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test1",
					Namespace: "foo",
				},
				InvolvedObject: api.ObjectReference{
					Namespace: "bar",
					Kind:      "Pod",
				},
			},
			false,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test2",
					Namespace: "aoeu-_-aoeu",
				},
				InvolvedObject: api.ObjectReference{
					Namespace: "aoeu-_-aoeu",
					Kind:      "Pod",
				},
			},
			false,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test3",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "v1",
					Kind:       "Node",
				},
			},
			true,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test4",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "v1",
					Kind:       "Namespace",
				},
			},
			true,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test5",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "extensions/v1beta1",
					Kind:       "NoKind",
					Namespace:  metav1.NamespaceDefault,
				},
			},
			true,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test6",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "extensions/v1beta1",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			false,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test7",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "extensions/v1beta1",
					Kind:       "Job",
					Namespace:  metav1.NamespaceDefault,
				},
			},
			true,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test8",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "other/v1beta1",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			false,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test9",
					Namespace: "foo",
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "other/v1beta1",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			true,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test10",
					Namespace: metav1.NamespaceDefault,
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "extensions",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			false,
		}, {
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test11",
					Namespace: "foo",
				},
				InvolvedObject: api.ObjectReference{
					// must register in v1beta1 to be true
					APIVersion: "extensions/v1beta1",
					Kind:       "Job",
					Namespace:  "foo",
				},
			},
			true,
		},
		{
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test12",
					Namespace: "foo",
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "other/v1beta1",
					Kind:       "FooBar",
					Namespace:  "bar",
				},
			},
			false,
		},
		{
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test13",
					Namespace: "",
				},
				InvolvedObject: api.ObjectReference{
					APIVersion: "other/v1beta1",
					Kind:       "FooBar",
					Namespace:  "bar",
				},
			},
			false,
		},
		{
			&api.Event{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test14",
					Namespace: "foo",
				},
				InvolvedObject: api.ObjectReference{
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
			t.Errorf("%v: expected %v, got %v", item.Event.Name, e, a)
		}
	}
}
