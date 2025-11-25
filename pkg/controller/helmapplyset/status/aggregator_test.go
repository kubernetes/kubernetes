/*
Copyright 2024 The Kubernetes Authors.

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

package status

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	apps "k8s.io/api/apps/v1"
	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
)

func TestDeploymentHealthChecker_CheckHealth(t *testing.T) {
	tests := []struct {
		name       string
		deployment *apps.Deployment
		expected   HealthStatus
	}{
		{
			name: "healthy deployment",
			deployment: &apps.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: apps.DeploymentSpec{
					Replicas: int32Ptr(3),
				},
				Status: apps.DeploymentStatus{
					ReadyReplicas:     3,
					UpdatedReplicas:   3,
					AvailableReplicas: 3,
				},
			},
			expected: HealthStatus{
				Healthy: true,
				Reason:  "AllReplicasReady",
				Message: "Deployment has 3/3 ready replicas",
			},
		},
		{
			name: "progressing deployment",
			deployment: &apps.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: apps.DeploymentSpec{
					Replicas: int32Ptr(3),
				},
				Status: apps.DeploymentStatus{
					ReadyReplicas:     2,
					UpdatedReplicas:   2,
					AvailableReplicas: 2,
				},
			},
			expected: HealthStatus{
				Healthy: false,
				Reason:  "Progressing",
				Message: "Deployment is updating: 2/3 replicas updated",
			},
		},
		{
			name: "degraded deployment",
			deployment: &apps.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: apps.DeploymentSpec{
					Replicas: int32Ptr(3),
				},
				Status: apps.DeploymentStatus{
					ReadyReplicas:     1,
					UpdatedReplicas:   3,
					AvailableReplicas: 1,
				},
			},
			expected: HealthStatus{
				Healthy: false,
				Reason:  "ReplicasNotReady",
				Message: "Deployment has 1/3 ready replicas",
			},
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	checker := &DeploymentHealthChecker{logger: logger}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := toUnstructured(tt.deployment)
			require.NoError(t, err)

			result, err := checker.CheckHealth(context.Background(), obj)
			require.NoError(t, err)

			assert.Equal(t, tt.expected.Healthy, result.Healthy)
			assert.Equal(t, tt.expected.Reason, result.Reason)
			assert.Contains(t, result.Message, tt.expected.Message)
		})
	}
}

func TestStatefulSetHealthChecker_CheckHealth(t *testing.T) {
	tests := []struct {
		name        string
		statefulSet *apps.StatefulSet
		expected    HealthStatus
	}{
		{
			name: "healthy statefulset",
			statefulSet: &apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-statefulset",
					Namespace: "default",
				},
				Spec: apps.StatefulSetSpec{
					Replicas: int32Ptr(3),
				},
				Status: apps.StatefulSetStatus{
					ReadyReplicas:   3,
					CurrentReplicas: 3,
				},
			},
			expected: HealthStatus{
				Healthy: true,
				Reason:  "AllReplicasReady",
				Message: "StatefulSet has 3/3 ready replicas",
			},
		},
		{
			name: "progressing statefulset",
			statefulSet: &apps.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-statefulset",
					Namespace: "default",
				},
				Spec: apps.StatefulSetSpec{
					Replicas: int32Ptr(3),
				},
				Status: apps.StatefulSetStatus{
					ReadyReplicas:   2,
					CurrentReplicas: 2,
				},
			},
			expected: HealthStatus{
				Healthy: false,
				Reason:  "Progressing",
				Message: "StatefulSet is updating: 2/3 replicas current",
			},
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	checker := &StatefulSetHealthChecker{logger: logger}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := toUnstructured(tt.statefulSet)
			require.NoError(t, err)

			result, err := checker.CheckHealth(context.Background(), obj)
			require.NoError(t, err)

			assert.Equal(t, tt.expected.Healthy, result.Healthy)
			assert.Equal(t, tt.expected.Reason, result.Reason)
			assert.Contains(t, result.Message, tt.expected.Message)
		})
	}
}

func TestDaemonSetHealthChecker_CheckHealth(t *testing.T) {
	tests := []struct {
		name      string
		daemonSet *apps.DaemonSet
		expected  HealthStatus
	}{
		{
			name: "healthy daemonset",
			daemonSet: &apps.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-daemonset",
					Namespace: "default",
				},
				Status: apps.DaemonSetStatus{
					DesiredNumberScheduled: 5,
					NumberReady:            5,
					NumberUnavailable:      0,
				},
			},
			expected: HealthStatus{
				Healthy: true,
				Reason:  "AllPodsReady",
				Message: "DaemonSet has 5/5 pods ready",
			},
		},
		{
			name: "unavailable pods",
			daemonSet: &apps.DaemonSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-daemonset",
					Namespace: "default",
				},
				Status: apps.DaemonSetStatus{
					DesiredNumberScheduled: 5,
					NumberReady:            3,
					NumberUnavailable:      2,
				},
			},
			expected: HealthStatus{
				Healthy: false,
				Reason:  "PodsUnavailable",
				Message: "DaemonSet has 2 unavailable pods",
			},
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	checker := &DaemonSetHealthChecker{logger: logger}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := toUnstructured(tt.daemonSet)
			require.NoError(t, err)

			result, err := checker.CheckHealth(context.Background(), obj)
			require.NoError(t, err)

			assert.Equal(t, tt.expected.Healthy, result.Healthy)
			assert.Equal(t, tt.expected.Reason, result.Reason)
			assert.Contains(t, result.Message, tt.expected.Message)
		})
	}
}

func TestServiceHealthChecker_CheckHealth(t *testing.T) {
	tests := []struct {
		name      string
		service   *v1.Service
		endpoints *v1.Endpoints
		expected  HealthStatus
	}{
		{
			name: "healthy service with endpoints",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-service",
					Namespace: "default",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"app": "test"},
				},
			},
			endpoints: &v1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-service",
					Namespace: "default",
				},
				Subsets: []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{
							{IP: "10.0.0.1"},
							{IP: "10.0.0.2"},
						},
					},
				},
			},
			expected: HealthStatus{
				Healthy: true,
				Reason:  "EndpointsReady",
				Message: "Service has 2 ready endpoint addresses",
			},
		},
		{
			name: "service without endpoints",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-service",
					Namespace: "default",
				},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"app": "test"},
				},
			},
			endpoints: nil, // No endpoints
			expected: HealthStatus{
				Healthy: false,
				Reason:  "NoEndpoints",
				Message: "Service has no endpoints",
			},
		},
		{
			name: "headless service",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-service",
					Namespace: "default",
				},
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone,
				},
			},
			expected: HealthStatus{
				Healthy: true,
				Reason:  "NoEndpointsRequired",
				Message: "Service does not require endpoints",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			client := fake.NewSimpleClientset()
			if tt.endpoints != nil {
				client.CoreV1().Endpoints("default").Create(context.Background(), tt.endpoints, metav1.CreateOptions{})
			}

			checker := &ServiceHealthChecker{
				kubeClient: client,
				logger:     logger,
			}

			obj, err := toUnstructured(tt.service)
			require.NoError(t, err)

			result, err := checker.CheckHealth(context.Background(), obj)
			require.NoError(t, err)

			assert.Equal(t, tt.expected.Healthy, result.Healthy)
			assert.Equal(t, tt.expected.Reason, result.Reason)
			assert.Contains(t, result.Message, tt.expected.Message)
		})
	}
}

func TestPVCHealthChecker_CheckHealth(t *testing.T) {
	tests := []struct {
		name     string
		pvc      *v1.PersistentVolumeClaim
		expected HealthStatus
	}{
		{
			name: "bound PVC",
			pvc: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pvc",
					Namespace: "default",
				},
				Status: v1.PersistentVolumeClaimStatus{
					Phase: v1.ClaimBound,
				},
			},
			expected: HealthStatus{
				Healthy: true,
				Reason:  "Bound",
				Message: "PVC is bound",
			},
		},
		{
			name: "pending PVC",
			pvc: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pvc",
					Namespace: "default",
				},
				Status: v1.PersistentVolumeClaimStatus{
					Phase: v1.ClaimPending,
				},
			},
			expected: HealthStatus{
				Healthy: false,
				Reason:  "Pending",
				Message: "PVC is pending binding",
			},
		},
		{
			name: "lost PVC",
			pvc: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pvc",
					Namespace: "default",
				},
				Status: v1.PersistentVolumeClaimStatus{
					Phase: v1.ClaimLost,
				},
			},
			expected: HealthStatus{
				Healthy: false,
				Reason:  "Lost",
				Message: "PVC is lost",
			},
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	checker := &PVCHealthChecker{logger: logger}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := toUnstructured(tt.pvc)
			require.NoError(t, err)

			result, err := checker.CheckHealth(context.Background(), obj)
			require.NoError(t, err)

			assert.Equal(t, tt.expected.Healthy, result.Healthy)
			assert.Equal(t, tt.expected.Reason, result.Reason)
			assert.Equal(t, tt.expected.Message, result.Message)
		})
	}
}

func TestJobHealthChecker_CheckHealth(t *testing.T) {
	tests := []struct {
		name     string
		job      *batch.Job
		expected HealthStatus
	}{
		{
			name: "completed job",
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
				},
				Spec: batch.JobSpec{
					Completions: int32Ptr(3),
				},
				Status: batch.JobStatus{
					Succeeded: 3,
					Failed:    0,
					Active:    0,
				},
			},
			expected: HealthStatus{
				Healthy: true,
				Reason:  "Completed",
				Message: "Job completed successfully: 3/3 completions",
			},
		},
		{
			name: "failed job",
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
				},
				Spec: batch.JobSpec{
					Completions: int32Ptr(3),
				},
				Status: batch.JobStatus{
					Succeeded: 1,
					Failed:    2,
					Active:    0,
				},
			},
			expected: HealthStatus{
				Healthy: false,
				Reason:  "Failed",
				Message: "Job failed: 2 failed pods",
			},
		},
		{
			name: "progressing job",
			job: &batch.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-job",
					Namespace: "default",
				},
				Spec: batch.JobSpec{
					Completions: int32Ptr(3),
				},
				Status: batch.JobStatus{
					Succeeded: 1,
					Failed:    0,
					Active:    2,
				},
			},
			expected: HealthStatus{
				Healthy: false,
				Reason:  "Progressing",
				Message: "Job is running: 2 active pods",
			},
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	checker := &JobHealthChecker{logger: logger}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj, err := toUnstructured(tt.job)
			require.NoError(t, err)

			result, err := checker.CheckHealth(context.Background(), obj)
			require.NoError(t, err)

			assert.Equal(t, tt.expected.Healthy, result.Healthy)
			assert.Equal(t, tt.expected.Reason, result.Reason)
			assert.Contains(t, result.Message, tt.expected.Message)
		})
	}
}

func TestGenericHealthChecker_CheckHealth(t *testing.T) {
	tests := []struct {
		name     string
		obj      map[string]interface{}
		expected HealthStatus
	}{
		{
			name: "resource with Ready=True condition",
			obj: map[string]interface{}{
				"status": map[string]interface{}{
					"conditions": []interface{}{
						map[string]interface{}{
							"type":    "Ready",
							"status":  "True",
							"reason":  "ResourceReady",
							"message": "Resource is ready",
						},
					},
				},
			},
			expected: HealthStatus{
				Healthy: true,
				Reason:  "ResourceReady",
				Message: "Resource is ready",
			},
		},
		{
			name: "resource with Ready=False condition",
			obj: map[string]interface{}{
				"status": map[string]interface{}{
					"conditions": []interface{}{
						map[string]interface{}{
							"type":    "Ready",
							"status":  "False",
							"reason":  "ResourceFailed",
							"message": "Resource failed",
						},
					},
				},
			},
			expected: HealthStatus{
				Healthy: false,
				Reason:  "ResourceFailed",
				Message: "Resource failed",
			},
		},
		{
			name: "resource with no conditions",
			obj: map[string]interface{}{
				"status": map[string]interface{}{},
			},
			expected: HealthStatus{
				Healthy: true, // Default to healthy if no conditions found (lenient)
				Reason:  "NoConditions",
			},
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	checker := &GenericHealthChecker{logger: logger}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			obj := &unstructured.Unstructured{Object: tt.obj}
			result, err := checker.CheckHealth(context.Background(), obj)
			require.NoError(t, err)

			assert.Equal(t, tt.expected.Healthy, result.Healthy)
			if tt.expected.Reason != "" {
				assert.Equal(t, tt.expected.Reason, result.Reason)
			}
			if tt.expected.Message != "" {
				assert.Equal(t, tt.expected.Message, result.Message)
			}
		})
	}
}

func TestReleaseHealth_computeAggregateStatus(t *testing.T) {
	tests := []struct {
		name           string
		resourceHealth map[string]HealthStatus
		expectedStatus string
	}{
		{
			name: "all healthy",
			resourceHealth: map[string]HealthStatus{
				"apps/v1/Deployment/default/dep1": {Healthy: true},
				"v1/Service/default/svc1":         {Healthy: true},
			},
			expectedStatus: "healthy",
		},
		{
			name: "some progressing",
			resourceHealth: map[string]HealthStatus{
				"apps/v1/Deployment/default/dep1": {Healthy: true},
				"apps/v1/Deployment/default/dep2": {Healthy: false, Reason: "Progressing"},
			},
			expectedStatus: "progressing",
		},
		{
			name: "some degraded",
			resourceHealth: map[string]HealthStatus{
				"apps/v1/Deployment/default/dep1": {Healthy: true},
				"apps/v1/Deployment/default/dep2": {Healthy: false, Reason: "ReplicasNotReady"},
			},
			expectedStatus: "degraded",
		},
		{
			name: "some failed",
			resourceHealth: map[string]HealthStatus{
				"apps/v1/Deployment/default/dep1": {Healthy: true},
				"batch/v1/Job/default/job1":       {Healthy: false, Reason: "Failed"},
			},
			expectedStatus: "failed",
		},
		{
			name:           "no resources",
			resourceHealth: map[string]HealthStatus{},
			expectedStatus: "unknown",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			health := &ReleaseHealth{
				ResourceHealth: tt.resourceHealth,
			}
			health.computeAggregateStatus()

			assert.Equal(t, tt.expectedStatus, health.OverallStatus)
			assert.Equal(t, len(tt.resourceHealth), health.TotalResources)
		})
	}
}

// Helper functions

func int32Ptr(i int32) *int32 {
	return &i
}

func toUnstructured(obj runtime.Object) (*unstructured.Unstructured, error) {
	unstructuredObj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	return &unstructured.Unstructured{Object: unstructuredObj}, nil
}
