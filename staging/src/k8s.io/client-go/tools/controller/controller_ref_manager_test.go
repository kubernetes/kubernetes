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

package controller

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

var (
	match    = func(metav1.Object) bool { return true }
	notMatch = func(metav1.Object) bool { return false }
)

type fixture struct {
	adopted  bool
	released bool

	expAdoption bool
	expRelease  bool
}

func (f *fixture) adopt(metav1.Object) error {
	f.adopted = true
	return nil
}

func (f *fixture) release(metav1.Object) error {
	f.released = true
	return nil
}

func newPod(owner metav1.Object) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: metav1.NamespaceDefault,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Image: "foo/bar",
				},
			},
		},
	}
	if owner != nil {
		pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(owner, v1beta1.SchemeGroupVersion.WithKind("Fake"))}
	}
	return pod
}

func TestClaimObject(t *testing.T) {
	type test struct {
		desc       string
		object     metav1.Object
		controller metav1.Object
		fixture    *fixture
		matchFunc  func(metav1.Object) bool

		exp    bool
		expErr error
	}

	tests := []test{
		{
			desc:       "Adopt orphan",
			controller: &v1.ReplicationController{},
			object:     newPod(nil),
			fixture:    &fixture{expAdoption: true},
			matchFunc:  match,

			exp:    true,
			expErr: nil,
		},
		func() test {
			controller := &v1.ReplicationController{}
			controller2 := &v1.ReplicationController{}
			controller.UID = types.UID("test")
			controller2.UID = types.UID("another-test")

			return test{
				desc:       "Do nothing with object owned by another controller",
				controller: controller,
				object:     newPod(controller2),
				fixture:    &fixture{expAdoption: false, expRelease: false},
				matchFunc:  match,

				exp:    false,
				expErr: nil,
			}
		}(),
		func() test {
			controller := &v1.ReplicationController{}
			controller.UID = types.UID("test")

			return test{
				desc:       "Do nothing with own object",
				controller: controller,
				object:     newPod(controller),
				fixture:    &fixture{expAdoption: false, expRelease: false},
				matchFunc:  match,

				exp:    true,
				expErr: nil,
			}
		}(),
		func() test {
			controller := &v1.ReplicationController{}
			controller.UID = types.UID("test")
			return test{
				desc:       "Release claimed pods when selector doesn't match",
				controller: controller,
				object:     newPod(controller),
				fixture:    &fixture{expAdoption: false, expRelease: true},
				matchFunc:  notMatch,

				exp:    false,
				expErr: nil,
			}
		}(),
	}
	for _, test := range tests {
		t.Run(test.desc, func(t *testing.T) {
			c := BaseControllerRefManager{
				Controller: test.controller,
			}
			got, err := c.ClaimObject(test.object, test.matchFunc, test.fixture.adopt, test.fixture.release)

			if got != test.exp {
				t.Errorf("ClaimObject() returned %t expected %t", got, test.exp)
			}
			if !reflect.DeepEqual(err, test.expErr) {
				t.Errorf("ClaimObject() returned error %s expected %s", err, test.expErr)
			}

			if test.fixture.adopted == true && test.fixture.expAdoption == false {
				t.Error("ClaimObject() expected to not adopt object but adopted it")
			}
			if test.fixture.adopted == false && test.fixture.expAdoption == true {
				t.Error("ClaimObject() expected to adopt object but not adopted it")
			}

			if test.fixture.released == true && test.fixture.expRelease == false {
				t.Error("ClaimObject() expected to not release object but released it")
			}
			if test.fixture.released == false && test.fixture.expRelease == true {
				t.Error("ClaimObject() expected to release object but not released it")
			}
		})
	}
}
