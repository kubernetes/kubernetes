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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

func TestConditionManager_computeReadyCondition(t *testing.T) {
	tests := []struct {
		name           string
		health         *ReleaseHealth
		expectedStatus metav1.ConditionStatus
		expectedReason string
	}{
		{
			name: "healthy release",
			health: &ReleaseHealth{
				OverallStatus:    "healthy",
				TotalResources:   5,
				HealthyResources: 5,
			},
			expectedStatus: metav1.ConditionTrue,
			expectedReason: "AllResourcesHealthy",
		},
		{
			name: "progressing release",
			health: &ReleaseHealth{
				OverallStatus:        "progressing",
				TotalResources:       5,
				HealthyResources:     3,
				ProgressingResources: 2,
			},
			expectedStatus: metav1.ConditionFalse,
			expectedReason: "ResourcesProgressing",
		},
		{
			name: "degraded release",
			health: &ReleaseHealth{
				OverallStatus:     "degraded",
				TotalResources:    5,
				HealthyResources:  3,
				DegradedResources: 2,
				ResourceHealth: map[string]HealthStatus{
					"apps/v1/Deployment/default/dep1": {Healthy: false, Reason: "ReplicasNotReady"},
					"v1/Service/default/svc1":         {Healthy: false, Reason: "NoEndpoints"},
				},
			},
			expectedStatus: metav1.ConditionFalse,
			expectedReason: "ResourcesDegraded",
		},
		{
			name: "failed release",
			health: &ReleaseHealth{
				OverallStatus:    "failed",
				TotalResources:   5,
				HealthyResources: 3,
				FailedResources:  2,
				ResourceHealth: map[string]HealthStatus{
					"batch/v1/Job/default/job1": {Healthy: false, Reason: "Failed"},
				},
			},
			expectedStatus: metav1.ConditionFalse,
			expectedReason: "ResourcesFailed",
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	cm := NewConditionManager(fake.NewSimpleClientset(), logger)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			condition := cm.computeReadyCondition(tt.health, metav1.Now())

			assert.Equal(t, "Ready", condition.Type)
			assert.Equal(t, tt.expectedStatus, condition.Status)
			assert.Equal(t, tt.expectedReason, condition.Reason)
			assert.NotEmpty(t, condition.Message)
		})
	}
}

func TestConditionManager_computeProgressingCondition(t *testing.T) {
	tests := []struct {
		name           string
		health         *ReleaseHealth
		expectedStatus metav1.ConditionStatus
	}{
		{
			name: "progressing release",
			health: &ReleaseHealth{
				ProgressingResources: 2,
				ResourceHealth: map[string]HealthStatus{
					"apps/v1/Deployment/default/dep1": {Healthy: false, Reason: "Progressing"},
				},
			},
			expectedStatus: metav1.ConditionTrue,
		},
		{
			name: "not progressing",
			health: &ReleaseHealth{
				ProgressingResources: 0,
			},
			expectedStatus: metav1.ConditionFalse,
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	cm := NewConditionManager(fake.NewSimpleClientset(), logger)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			condition := cm.computeProgressingCondition(tt.health, metav1.Now())

			assert.Equal(t, "Progressing", condition.Type)
			assert.Equal(t, tt.expectedStatus, condition.Status)
		})
	}
}

func TestConditionManager_computeDegradedCondition(t *testing.T) {
	tests := []struct {
		name           string
		health         *ReleaseHealth
		expectedStatus metav1.ConditionStatus
	}{
		{
			name: "degraded release",
			health: &ReleaseHealth{
				DegradedResources: 2,
				ResourceHealth: map[string]HealthStatus{
					"apps/v1/Deployment/default/dep1": {Healthy: false, Reason: "ReplicasNotReady"},
				},
			},
			expectedStatus: metav1.ConditionTrue,
		},
		{
			name: "healthy release",
			health: &ReleaseHealth{
				DegradedResources: 0,
				FailedResources:   0,
			},
			expectedStatus: metav1.ConditionFalse,
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	cm := NewConditionManager(fake.NewSimpleClientset(), logger)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			condition := cm.computeDegradedCondition(tt.health, metav1.Now())

			assert.Equal(t, "Degraded", condition.Type)
			assert.Equal(t, tt.expectedStatus, condition.Status)
		})
	}
}

func TestConditionManager_computeAvailableCondition(t *testing.T) {
	tests := []struct {
		name           string
		health         *ReleaseHealth
		expectedStatus metav1.ConditionStatus
	}{
		{
			name: "available (50% healthy)",
			health: &ReleaseHealth{
				TotalResources:   10,
				HealthyResources: 5,
			},
			expectedStatus: metav1.ConditionTrue,
		},
		{
			name: "available (more than 50% healthy)",
			health: &ReleaseHealth{
				TotalResources:   10,
				HealthyResources: 8,
			},
			expectedStatus: metav1.ConditionTrue,
		},
		{
			name: "not available (less than 50% healthy)",
			health: &ReleaseHealth{
				TotalResources:   10,
				HealthyResources: 4,
			},
			expectedStatus: metav1.ConditionFalse,
		},
		{
			name: "no resources",
			health: &ReleaseHealth{
				TotalResources:   0,
				HealthyResources: 0,
			},
			expectedStatus: metav1.ConditionUnknown,
		},
	}

	logger, _ := ktesting.NewTestContext(t)
	cm := NewConditionManager(fake.NewSimpleClientset(), logger)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			condition := cm.computeAvailableCondition(tt.health, metav1.Now())

			assert.Equal(t, "Available", condition.Type)
			assert.Equal(t, tt.expectedStatus, condition.Status)
		})
	}
}

func TestConditionManager_UpdateConditions(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	// Create parent Secret
	parentSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      parent.ParentSecretNamePrefix + "test-release",
			Namespace: "default",
			Labels: map[string]string{
				parent.ApplySetParentIDLabel: "applyset-test-id-v1",
			},
		},
		Type: v1.SecretTypeOpaque,
	}

	client := fake.NewSimpleClientset(parentSecret)
	cm := NewConditionManager(client, logger)

	health := &ReleaseHealth{
		ReleaseName:      "test-release",
		Namespace:        "default",
		OverallStatus:    "healthy",
		TotalResources:   3,
		HealthyResources: 3,
		ResourceHealth: map[string]HealthStatus{
			"apps/v1/Deployment/default/dep1": {Healthy: true},
			"v1/Service/default/svc1":         {Healthy: true},
			"v1/ConfigMap/default/cm1":        {Healthy: true},
		},
	}

	err := cm.UpdateConditions(ctx, "test-release", "default", health)
	require.NoError(t, err)

	// Verify conditions were updated
	updatedSecret, err := client.CoreV1().Secrets("default").Get(ctx, parent.ParentSecretNamePrefix+"test-release", metav1.GetOptions{})
	require.NoError(t, err)

	// Check Ready condition
	readyJSON, ok := updatedSecret.Annotations["status.conditions.ready"]
	require.True(t, ok, "Ready condition should be present")

	var readyCondition metav1.Condition
	err = json.Unmarshal([]byte(readyJSON), &readyCondition)
	require.NoError(t, err)
	assert.Equal(t, "Ready", readyCondition.Type)
	assert.Equal(t, metav1.ConditionTrue, readyCondition.Status)
	assert.Equal(t, "AllResourcesHealthy", readyCondition.Reason)
}

func TestConditionManager_mergeConditions(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cm := NewConditionManager(fake.NewSimpleClientset(), logger)

	now := metav1.Now()
	earlier := metav1.Time{Time: now.Add(-3600 * 1e9)} // 1 hour ago

	existing := []metav1.Condition{
		{
			Type:               "Ready",
			Status:             metav1.ConditionTrue,
			LastTransitionTime: earlier,
			Reason:             "AllResourcesHealthy",
			Message:            "All resources healthy",
		},
	}

	new := []metav1.Condition{
		{
			Type:               "Ready",
			Status:             metav1.ConditionTrue,
			LastTransitionTime: now,
			Reason:             "AllResourcesHealthy",
			Message:            "All resources healthy",
		},
	}

	merged := cm.mergeConditions(existing, new)

	// Should preserve lastTransitionTime since status and reason unchanged
	assert.Equal(t, 1, len(merged))
	assert.Equal(t, earlier, merged[0].LastTransitionTime, "Should preserve old lastTransitionTime")

	// Test with changed status
	new[0].Status = metav1.ConditionFalse
	new[0].Reason = "ResourcesDegraded"
	merged = cm.mergeConditions(existing, new)
	assert.Equal(t, now, merged[0].LastTransitionTime, "Should update lastTransitionTime when status changes")
}

func TestConditionManager_GetConditions(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	// Create parent Secret with conditions
	readyCondition := metav1.Condition{
		Type:               "Ready",
		Status:             metav1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             "AllResourcesHealthy",
		Message:            "All resources healthy",
	}
	readyJSON, _ := json.Marshal(readyCondition)

	parentSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      parent.ParentSecretNamePrefix + "test-release",
			Namespace: "default",
			Annotations: map[string]string{
				"status.conditions.ready": string(readyJSON),
			},
		},
		Type: v1.SecretTypeOpaque,
	}

	client := fake.NewSimpleClientset(parentSecret)
	cm := NewConditionManager(client, logger)

	conditions, err := cm.GetConditions(ctx, "test-release", "default")
	require.NoError(t, err)
	require.Equal(t, 1, len(conditions))
	assert.Equal(t, "Ready", conditions[0].Type)
	assert.Equal(t, metav1.ConditionTrue, conditions[0].Status)
}

func TestConditionManager_GetConditions_NotFound(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	cm := NewConditionManager(client, logger)

	_, err := cm.GetConditions(ctx, "test-release", "default")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}
