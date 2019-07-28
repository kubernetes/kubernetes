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

package aws

import (
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
)

func TestMapToAWSInstanceIDs(t *testing.T) {
	tests := []struct {
		Kubernetes  KubernetesInstanceID
		Aws         InstanceID
		ExpectError bool
	}{
		{
			Kubernetes: "aws:///us-east-1a/i-12345678",
			Aws:        "i-12345678",
		},
		{
			Kubernetes: "aws:////i-12345678",
			Aws:        "i-12345678",
		},
		{
			Kubernetes: "i-12345678",
			Aws:        "i-12345678",
		},
		{
			Kubernetes: "aws:///us-east-1a/i-12345678abcdef01",
			Aws:        "i-12345678abcdef01",
		},
		{
			Kubernetes: "aws:////i-12345678abcdef01",
			Aws:        "i-12345678abcdef01",
		},
		{
			Kubernetes: "i-12345678abcdef01",
			Aws:        "i-12345678abcdef01",
		},
		{
			Kubernetes:  "vol-123456789",
			ExpectError: true,
		},
		{
			Kubernetes:  "aws:///us-east-1a/vol-12345678abcdef01",
			ExpectError: true,
		},
		{
			Kubernetes:  "aws://accountid/us-east-1a/vol-12345678abcdef01",
			ExpectError: true,
		},
		{
			Kubernetes:  "aws:///us-east-1a/vol-12345678abcdef01/suffix",
			ExpectError: true,
		},
		{
			Kubernetes:  "",
			ExpectError: true,
		},
	}

	for _, test := range tests {
		awsID, err := test.Kubernetes.MapToAWSInstanceID()
		if err != nil {
			if !test.ExpectError {
				t.Errorf("unexpected error parsing %s: %v", test.Kubernetes, err)
			}
		} else {
			if test.ExpectError {
				t.Errorf("expected error parsing %s", test.Kubernetes)
			} else if test.Aws != awsID {
				t.Errorf("unexpected value parsing %s, got %s", test.Kubernetes, awsID)
			}
		}
	}

	for _, test := range tests {
		node := &v1.Node{}
		node.Spec.ProviderID = string(test.Kubernetes)

		awsInstanceIds, err := mapToAWSInstanceIDs([]*v1.Node{node})
		if err != nil {
			if !test.ExpectError {
				t.Errorf("unexpected error parsing %s: %v", test.Kubernetes, err)
			}
		} else {
			if test.ExpectError {
				t.Errorf("expected error parsing %s", test.Kubernetes)
			} else if len(awsInstanceIds) != 1 {
				t.Errorf("unexpected value parsing %s, got %s", test.Kubernetes, awsInstanceIds)
			} else if awsInstanceIds[0] != test.Aws {
				t.Errorf("unexpected value parsing %s, got %s", test.Kubernetes, awsInstanceIds)
			}
		}

		awsInstanceIds = mapToAWSInstanceIDsTolerant([]*v1.Node{node})
		if test.ExpectError {
			if len(awsInstanceIds) != 0 {
				t.Errorf("unexpected results parsing %s: %s", test.Kubernetes, awsInstanceIds)
			}
		} else {
			if len(awsInstanceIds) != 1 {
				t.Errorf("unexpected value parsing %s, got %s", test.Kubernetes, awsInstanceIds)
			} else if awsInstanceIds[0] != test.Aws {
				t.Errorf("unexpected value parsing %s, got %s", test.Kubernetes, awsInstanceIds)
			}
		}
	}
}

func TestSnapshotMeetsCriteria(t *testing.T) {
	snapshot := &allInstancesSnapshot{timestamp: time.Now().Add(-3601 * time.Second)}

	if !snapshot.MeetsCriteria(cacheCriteria{}) {
		t.Errorf("Snapshot should always meet empty criteria")
	}

	if snapshot.MeetsCriteria(cacheCriteria{MaxAge: time.Hour}) {
		t.Errorf("Snapshot did not honor MaxAge")
	}

	if snapshot.MeetsCriteria(cacheCriteria{HasInstances: []InstanceID{InstanceID("i-12345678")}}) {
		t.Errorf("Snapshot did not honor HasInstances with missing instances")
	}

	snapshot.instances = make(map[InstanceID]*ec2.Instance)
	snapshot.instances[InstanceID("i-12345678")] = &ec2.Instance{}

	if !snapshot.MeetsCriteria(cacheCriteria{HasInstances: []InstanceID{InstanceID("i-12345678")}}) {
		t.Errorf("Snapshot did not honor HasInstances with matching instances")
	}

	if snapshot.MeetsCriteria(cacheCriteria{HasInstances: []InstanceID{InstanceID("i-12345678"), InstanceID("i-00000000")}}) {
		t.Errorf("Snapshot did not honor HasInstances with partially matching instances")
	}
}

func TestOlderThan(t *testing.T) {
	t1 := time.Now()
	t2 := t1.Add(time.Second)

	s1 := &allInstancesSnapshot{timestamp: t1}
	s2 := &allInstancesSnapshot{timestamp: t2}

	assert.True(t, s1.olderThan(s2), "s1 should be olderThan s2")
	assert.False(t, s2.olderThan(s1), "s2 not should be olderThan s1")
	assert.False(t, s1.olderThan(s1), "s1 not should be olderThan itself")
}

func TestSnapshotFindInstances(t *testing.T) {
	snapshot := &allInstancesSnapshot{}

	snapshot.instances = make(map[InstanceID]*ec2.Instance)
	{
		id := InstanceID("i-12345678")
		snapshot.instances[id] = &ec2.Instance{InstanceId: id.awsString()}
	}
	{
		id := InstanceID("i-23456789")
		snapshot.instances[id] = &ec2.Instance{InstanceId: id.awsString()}
	}

	instances := snapshot.FindInstances([]InstanceID{InstanceID("i-12345678"), InstanceID("i-23456789"), InstanceID("i-00000000")})
	if len(instances) != 2 {
		t.Errorf("findInstances returned %d results, expected 2", len(instances))
	}

	for _, id := range []InstanceID{InstanceID("i-12345678"), InstanceID("i-23456789")} {
		i := instances[id]
		if i == nil {
			t.Errorf("findInstances did not return %s", id)
			continue
		}
		if aws.StringValue(i.InstanceId) != string(id) {
			t.Errorf("findInstances did not return expected instanceId for %s", id)
		}
		if i != snapshot.instances[id] {
			t.Errorf("findInstances did not return expected instance (reference equality) for %s", id)
		}
	}
}
