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

package replication

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
)

var (
	imagePullBackOff v1.ReplicationControllerConditionType = "ImagePullBackOff"

	condImagePullBackOff = func() v1.ReplicationControllerCondition {
		return v1.ReplicationControllerCondition{
			Type:   imagePullBackOff,
			Status: v1.ConditionTrue,
			Reason: "NonExistentImage",
		}
	}

	condReplicaFailure = func() v1.ReplicationControllerCondition {
		return v1.ReplicationControllerCondition{
			Type:   v1.ReplicationControllerReplicaFailure,
			Status: v1.ConditionTrue,
			Reason: "OtherFailure",
		}
	}

	condReplicaFailure2 = func() v1.ReplicationControllerCondition {
		return v1.ReplicationControllerCondition{
			Type:   v1.ReplicationControllerReplicaFailure,
			Status: v1.ConditionTrue,
			Reason: "AnotherFailure",
		}
	}

	status = func() *v1.ReplicationControllerStatus {
		return &v1.ReplicationControllerStatus{
			Conditions: []v1.ReplicationControllerCondition{condReplicaFailure()},
		}
	}
)

func TestGetCondition(t *testing.T) {
	exampleStatus := status()

	tests := []struct {
		name string

		status     v1.ReplicationControllerStatus
		condType   v1.ReplicationControllerConditionType
		condStatus v1.ConditionStatus
		condReason string

		expected bool
	}{
		{
			name: "condition exists",

			status:   *exampleStatus,
			condType: v1.ReplicationControllerReplicaFailure,

			expected: true,
		},
		{
			name: "condition does not exist",

			status:   *exampleStatus,
			condType: imagePullBackOff,

			expected: false,
		},
	}

	for _, test := range tests {
		cond := GetCondition(test.status, test.condType)
		exists := cond != nil
		if exists != test.expected {
			t.Errorf("%s: expected condition to exist: %t, got: %t", test.name, test.expected, exists)
		}
	}
}

func TestSetCondition(t *testing.T) {
	tests := []struct {
		name string

		status *v1.ReplicationControllerStatus
		cond   v1.ReplicationControllerCondition

		expectedStatus *v1.ReplicationControllerStatus
	}{
		{
			name: "set for the first time",

			status: &v1.ReplicationControllerStatus{},
			cond:   condReplicaFailure(),

			expectedStatus: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condReplicaFailure()}},
		},
		{
			name: "simple set",

			status: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condImagePullBackOff()}},
			cond:   condReplicaFailure(),

			expectedStatus: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condImagePullBackOff(), condReplicaFailure()}},
		},
		{
			name: "overwrite",

			status: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condReplicaFailure()}},
			cond:   condReplicaFailure2(),

			expectedStatus: &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condReplicaFailure2()}},
		},
	}

	for _, test := range tests {
		SetCondition(test.status, test.cond)
		if !reflect.DeepEqual(test.status, test.expectedStatus) {
			t.Errorf("%s: expected status: %v, got: %v", test.name, test.expectedStatus, test.status)
		}
	}
}

func TestRemoveCondition(t *testing.T) {
	tests := []struct {
		name string

		status   *v1.ReplicationControllerStatus
		condType v1.ReplicationControllerConditionType

		expectedStatus *v1.ReplicationControllerStatus
	}{
		{
			name: "remove from empty status",

			status:   &v1.ReplicationControllerStatus{},
			condType: v1.ReplicationControllerReplicaFailure,

			expectedStatus: &v1.ReplicationControllerStatus{},
		},
		{
			name: "simple remove",

			status:   &v1.ReplicationControllerStatus{Conditions: []v1.ReplicationControllerCondition{condReplicaFailure()}},
			condType: v1.ReplicationControllerReplicaFailure,

			expectedStatus: &v1.ReplicationControllerStatus{},
		},
		{
			name: "doesn't remove anything",

			status:   status(),
			condType: imagePullBackOff,

			expectedStatus: status(),
		},
	}

	for _, test := range tests {
		RemoveCondition(test.status, test.condType)
		if !reflect.DeepEqual(test.status, test.expectedStatus) {
			t.Errorf("%s: expected status: %v, got: %v", test.name, test.expectedStatus, test.status)
		}
	}
}
