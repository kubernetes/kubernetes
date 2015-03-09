/*
Copyright 2014 Google Inc. All rights reserved.

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

package rest

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func TestCheckGeneratedNameError(t *testing.T) {
	expect := errors.NewNotFound("foo", "bar")
	if err := CheckGeneratedNameError(Services, expect, &api.Pod{}); err != expect {
		t.Errorf("NotFoundError should be ignored: %v", err)
	}

	expect = errors.NewAlreadyExists("foo", "bar")
	if err := CheckGeneratedNameError(Services, expect, &api.Pod{}); err != expect {
		t.Errorf("AlreadyExists should be returned when no GenerateName field: %v", err)
	}

	expect = errors.NewAlreadyExists("foo", "bar")
	if err := CheckGeneratedNameError(Services, expect, &api.Pod{ObjectMeta: api.ObjectMeta{GenerateName: "foo"}}); err == nil || !errors.IsServerTimeout(err) {
		t.Errorf("expected try again later error: %v", err)
	}
}

func TestBeforeCreate(t *testing.T) {
	failures := []runtime.Object{
		&api.Service{},
		&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: "#$%%invalid",
			},
		},
		&api.Service{
			ObjectMeta: api.ObjectMeta{
				Name:      "##&*(&invalid",
				Namespace: api.NamespaceDefault,
			},
		},
	}
	for _, test := range failures {
		ctx := api.NewDefaultContext()
		err := BeforeCreate(Services, ctx, test)
		if err == nil {
			t.Errorf("unexpected non-error for %v", test)
		}
	}

	obj := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: api.ReplicationControllerSpec{
			Selector: map[string]string{"name": "foo"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"name": "foo",
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:            "foo",
							Image:           "foo",
							ImagePullPolicy: api.PullAlways,
						},
					},
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSDefault,
				},
			},
		},
		Status: api.ReplicationControllerStatus{
			Replicas: 3,
		},
	}
	ctx := api.NewDefaultContext()
	err := BeforeCreate(ReplicationControllers, ctx, obj)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(obj.Status, api.ReplicationControllerStatus{}) {
		t.Errorf("status was not cleared as expected.")
	}
	if obj.Name != "foo" || obj.Namespace != api.NamespaceDefault {
		t.Errorf("unexpected object metadata: %v", obj.ObjectMeta)
	}

	obj.Spec.Replicas = -1
	if err := BeforeCreate(ReplicationControllers, ctx, obj); err == nil {
		t.Errorf("unexpected non-error for invalid replication controller.")
	}
}
