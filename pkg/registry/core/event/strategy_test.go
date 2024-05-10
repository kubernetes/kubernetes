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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/apis/core"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

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
