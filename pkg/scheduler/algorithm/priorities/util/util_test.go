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

package util

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGetControllerRef(t *testing.T) {
	fakeBlockOwnerDeletion := true
	fakeFalseController := false
	fakeTrueController := true
	fakeEmptyOwnerReference := metav1.OwnerReference{}

	tds := []struct {
		name        string
		pod         v1.Pod
		expectedNil bool
		expectedOR  metav1.OwnerReference
	}{
		{
			"ownerreference_not_exist",
			v1.Pod{},
			true,
			fakeEmptyOwnerReference,
		},
		{
			"ownerreference_controller_is_nil",
			v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion:         "extensions/v1beta1",
							Kind:               "ReplicaSet",
							Name:               "or-unit-test-5b9cffccff",
							UID:                "a46372ea-b254-11e7-8373-fa163e25bfb5",
							BlockOwnerDeletion: &fakeBlockOwnerDeletion,
						},
					},
				},
			},
			true,
			fakeEmptyOwnerReference,
		},
		{
			"ownerreference_controller_is_false",
			v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion:         "extensions/v1beta1",
							Kind:               "ReplicaSet",
							Name:               "or-unit-test-5b9cffccff",
							UID:                "a46372ea-b254-11e7-8373-fa163e25bfb5",
							Controller:         &fakeFalseController,
							BlockOwnerDeletion: &fakeBlockOwnerDeletion,
						},
					},
				},
			},
			true,
			fakeEmptyOwnerReference,
		},
		{
			"ownerreference_controller_is_true",
			v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion:         "extensions/v1beta1",
							Kind:               "ReplicaSet",
							Name:               "or-unit-test-5b9cffccff",
							UID:                "a46372ea-b254-11e7-8373-fa163e25bfb5",
							BlockOwnerDeletion: &fakeBlockOwnerDeletion,
							Controller:         &fakeTrueController,
						},
					},
				},
			},
			false,
			metav1.OwnerReference{
				APIVersion:         "extensions/v1beta1",
				Kind:               "ReplicaSet",
				Name:               "or-unit-test-5b9cffccff",
				UID:                "a46372ea-b254-11e7-8373-fa163e25bfb5",
				BlockOwnerDeletion: &fakeBlockOwnerDeletion,
				Controller:         &fakeTrueController,
			},
		},
	}

	for _, td := range tds {
		realOR := GetControllerRef(&td.pod)
		if td.expectedNil {
			assert.Nilf(t, realOR, "Failed to test: %s", td.name)
		} else {
			assert.Equalf(t, &td.expectedOR, realOR, "Failed to test: %s", td.name)
		}
	}
}
