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

package deployment

import (
	"testing"
	"time"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/controller/deployment/util"
)

func newDeploymentStatus(replicas, updatedReplicas, availableReplicas int32) extensions.DeploymentStatus {
	return extensions.DeploymentStatus{
		Replicas:          replicas,
		UpdatedReplicas:   updatedReplicas,
		AvailableReplicas: availableReplicas,
	}
}

// assumes the retuned deployment is always observed - not needed to be tested here.
func currentDeployment(pds *int32, replicas, statusReplicas, updatedReplicas, availableReplicas int32, conditions []extensions.DeploymentCondition) *extensions.Deployment {
	d := &extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: "progress-test",
		},
		Spec: extensions.DeploymentSpec{
			ProgressDeadlineSeconds: pds,
			Replicas:                &replicas,
			Strategy: extensions.DeploymentStrategy{
				Type: extensions.RecreateDeploymentStrategyType,
			},
		},
		Status: newDeploymentStatus(statusReplicas, updatedReplicas, availableReplicas),
	}
	d.Status.Conditions = conditions
	return d
}

func TestRequeueStuckDeployment(t *testing.T) {
	pds := int32(60)
	failed := []extensions.DeploymentCondition{
		{
			Type:   extensions.DeploymentProgressing,
			Status: v1.ConditionFalse,
			Reason: util.TimedOutReason,
		},
	}
	stuck := []extensions.DeploymentCondition{
		{
			Type:           extensions.DeploymentProgressing,
			Status:         v1.ConditionTrue,
			LastUpdateTime: metav1.Date(2017, 2, 15, 18, 49, 00, 00, time.UTC),
		},
	}

	tests := []struct {
		name     string
		d        *extensions.Deployment
		status   extensions.DeploymentStatus
		nowFn    func() time.Time
		expected time.Duration
	}{
		{
			name:     "no progressDeadlineSeconds specified",
			d:        currentDeployment(nil, 4, 3, 3, 2, nil),
			status:   newDeploymentStatus(3, 3, 2),
			expected: time.Duration(-1),
		},
		{
			name:     "no progressing condition found",
			d:        currentDeployment(&pds, 4, 3, 3, 2, nil),
			status:   newDeploymentStatus(3, 3, 2),
			expected: time.Duration(-1),
		},
		{
			name:     "complete deployment does not need to be requeued",
			d:        currentDeployment(&pds, 3, 3, 3, 3, nil),
			status:   newDeploymentStatus(3, 3, 3),
			expected: time.Duration(-1),
		},
		{
			name:     "already failed deployment does not need to be requeued",
			d:        currentDeployment(&pds, 3, 3, 3, 0, failed),
			status:   newDeploymentStatus(3, 3, 0),
			expected: time.Duration(-1),
		},
		{
			name:     "stuck deployment - 30s",
			d:        currentDeployment(&pds, 3, 3, 3, 1, stuck),
			status:   newDeploymentStatus(3, 3, 1),
			nowFn:    func() time.Time { return metav1.Date(2017, 2, 15, 18, 49, 30, 00, time.UTC).Time },
			expected: 30 * time.Second,
		},
		{
			name:     "stuck deployment - 1s",
			d:        currentDeployment(&pds, 3, 3, 3, 1, stuck),
			status:   newDeploymentStatus(3, 3, 1),
			nowFn:    func() time.Time { return metav1.Date(2017, 2, 15, 18, 49, 59, 00, time.UTC).Time },
			expected: 1 * time.Second,
		},
		{
			name:     "failed deployment - less than a second => now",
			d:        currentDeployment(&pds, 3, 3, 3, 1, stuck),
			status:   newDeploymentStatus(3, 3, 1),
			nowFn:    func() time.Time { return metav1.Date(2017, 2, 15, 18, 49, 59, 1, time.UTC).Time },
			expected: time.Duration(0),
		},
		{
			name:     "failed deployment - now",
			d:        currentDeployment(&pds, 3, 3, 3, 1, stuck),
			status:   newDeploymentStatus(3, 3, 1),
			nowFn:    func() time.Time { return metav1.Date(2017, 2, 15, 18, 50, 00, 00, time.UTC).Time },
			expected: time.Duration(0),
		},
		{
			name:     "failed deployment - 1s after deadline",
			d:        currentDeployment(&pds, 3, 3, 3, 1, stuck),
			status:   newDeploymentStatus(3, 3, 1),
			nowFn:    func() time.Time { return metav1.Date(2017, 2, 15, 18, 50, 01, 00, time.UTC).Time },
			expected: time.Duration(0),
		},
		{
			name:     "failed deployment - 60s after deadline",
			d:        currentDeployment(&pds, 3, 3, 3, 1, stuck),
			status:   newDeploymentStatus(3, 3, 1),
			nowFn:    func() time.Time { return metav1.Date(2017, 2, 15, 18, 51, 00, 00, time.UTC).Time },
			expected: time.Duration(0),
		},
	}

	dc := &DeploymentController{
		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "doesnt-matter"),
	}
	dc.enqueueDeployment = dc.enqueue

	for _, test := range tests {
		if test.nowFn != nil {
			nowFn = test.nowFn
		}
		got := dc.requeueStuckDeployment(test.d, test.status)
		if got != test.expected {
			t.Errorf("%s: got duration: %v, expected duration: %v", test.name, got, test.expected)
		}
	}
}
