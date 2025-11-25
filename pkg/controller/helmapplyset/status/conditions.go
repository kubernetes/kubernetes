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
	"encoding/json"
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
)

// ConditionManager manages conditions on ApplySet parent objects
type ConditionManager struct {
	kubeClient kubernetes.Interface
	logger     klog.Logger
}

// NewConditionManager creates a new condition manager
func NewConditionManager(
	kubeClient kubernetes.Interface,
	logger klog.Logger,
) *ConditionManager {
	return &ConditionManager{
		kubeClient: kubeClient,
		logger:     logger,
	}
}

// UpdateConditions updates conditions on ApplySet parent based on health status
func (cm *ConditionManager) UpdateConditions(
	ctx context.Context,
	releaseName, namespace string,
	health *ReleaseHealth,
) error {
	logger := klog.FromContext(ctx)
	logger.Info("Updating conditions for Helm release",
		"release", releaseName,
		"namespace", namespace,
		"overallStatus", health.OverallStatus)

	// Compute conditions from health status
	conditions := cm.computeConditions(health)

	// Update parent Secret with conditions
	return cm.updateParentConditions(ctx, releaseName, namespace, conditions)
}

// computeConditions computes standard Kubernetes conditions from health status
func (cm *ConditionManager) computeConditions(health *ReleaseHealth) []metav1.Condition {
	now := metav1.Now()
	conditions := make([]metav1.Condition, 0, 4)

	// Ready Condition
	readyCondition := cm.computeReadyCondition(health, now)
	conditions = append(conditions, readyCondition)

	// Progressing Condition
	progressingCondition := cm.computeProgressingCondition(health, now)
	conditions = append(conditions, progressingCondition)

	// Degraded Condition
	degradedCondition := cm.computeDegradedCondition(health, now)
	conditions = append(conditions, degradedCondition)

	// Available Condition
	availableCondition := cm.computeAvailableCondition(health, now)
	conditions = append(conditions, availableCondition)

	return conditions
}

// computeReadyCondition computes the Ready condition
func (cm *ConditionManager) computeReadyCondition(health *ReleaseHealth, now metav1.Time) metav1.Condition {
	condition := metav1.Condition{
		Type:               "Ready",
		LastTransitionTime: now,
	}

	switch health.OverallStatus {
	case "healthy":
		condition.Status = metav1.ConditionTrue
		condition.Reason = "AllResourcesHealthy"
		condition.Message = fmt.Sprintf("All %d resources are healthy", health.TotalResources)

	case "progressing":
		condition.Status = metav1.ConditionFalse
		condition.Reason = "ResourcesProgressing"
		condition.Message = fmt.Sprintf("%d resources are progressing, %d healthy",
			health.ProgressingResources, health.HealthyResources)

	case "degraded":
		condition.Status = metav1.ConditionFalse
		condition.Reason = "ResourcesDegraded"
		condition.Message = fmt.Sprintf("%d resources are degraded: %s",
			health.DegradedResources, strings.Join(health.GetUnhealthyResourceNames(), ", "))

	case "failed":
		condition.Status = metav1.ConditionFalse
		condition.Reason = "ResourcesFailed"
		condition.Message = fmt.Sprintf("%d resources failed: %s",
			health.FailedResources, strings.Join(health.GetFailedResourceNames(), ", "))

	default:
		condition.Status = metav1.ConditionUnknown
		condition.Reason = "StatusUnknown"
		condition.Message = fmt.Sprintf("Release status is unknown: %s", health.OverallStatus)
	}

	return condition
}

// computeProgressingCondition computes the Progressing condition
func (cm *ConditionManager) computeProgressingCondition(health *ReleaseHealth, now metav1.Time) metav1.Condition {
	condition := metav1.Condition{
		Type:               "Progressing",
		LastTransitionTime: now,
	}

	if health.ProgressingResources > 0 {
		condition.Status = metav1.ConditionTrue
		condition.Reason = "RolloutInProgress"
		condition.Message = fmt.Sprintf("%d resources are progressing: %s",
			health.ProgressingResources, strings.Join(health.GetProgressingResourceNames(), ", "))
	} else {
		condition.Status = metav1.ConditionFalse
		condition.Reason = "NoRolloutInProgress"
		condition.Message = "No rollout is in progress"
	}

	return condition
}

// computeDegradedCondition computes the Degraded condition
func (cm *ConditionManager) computeDegradedCondition(health *ReleaseHealth, now metav1.Time) metav1.Condition {
	condition := metav1.Condition{
		Type:               "Degraded",
		LastTransitionTime: now,
	}

	if health.DegradedResources > 0 || health.FailedResources > 0 {
		condition.Status = metav1.ConditionTrue
		condition.Reason = "ResourcesUnhealthy"
		unhealthyResources := append(
			health.GetUnhealthyResourceNames(),
			health.GetFailedResourceNames()...,
		)
		condition.Message = fmt.Sprintf("%d resources are unhealthy: %s",
			health.DegradedResources+health.FailedResources,
			strings.Join(unhealthyResources, ", "))
	} else {
		condition.Status = metav1.ConditionFalse
		condition.Reason = "AllResourcesHealthy"
		condition.Message = "All resources are healthy"
	}

	return condition
}

// computeAvailableCondition computes the Available condition
func (cm *ConditionManager) computeAvailableCondition(health *ReleaseHealth, now metav1.Time) metav1.Condition {
	condition := metav1.Condition{
		Type:               "Available",
		LastTransitionTime: now,
	}

	// Consider available if at least 50% of resources are healthy
	minimumAvailability := health.TotalResources / 2
	if health.TotalResources == 0 {
		condition.Status = metav1.ConditionUnknown
		condition.Reason = "NoResources"
		condition.Message = "No resources found"
	} else if health.HealthyResources >= minimumAvailability {
		condition.Status = metav1.ConditionTrue
		condition.Reason = "MinimumAvailabilityMet"
		condition.Message = fmt.Sprintf("%d/%d resources are healthy (minimum: %d)",
			health.HealthyResources, health.TotalResources, minimumAvailability)
	} else {
		condition.Status = metav1.ConditionFalse
		condition.Reason = "BelowMinimumAvailability"
		condition.Message = fmt.Sprintf("Only %d/%d resources are healthy (minimum: %d)",
			health.HealthyResources, health.TotalResources, minimumAvailability)
	}

	return condition
}

// updateParentConditions updates conditions on the ApplySet parent Secret
func (cm *ConditionManager) updateParentConditions(
	ctx context.Context,
	releaseName, namespace string,
	conditions []metav1.Condition,
) error {
	logger := klog.FromContext(ctx)
	parentName := fmt.Sprintf("%s%s", parent.ParentSecretNamePrefix, releaseName)

	// Use retry with exponential backoff for update
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// Get current parent Secret
		secret, err := cm.kubeClient.CoreV1().Secrets(namespace).Get(ctx, parentName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			logger.V(4).Info("Parent Secret not found, skipping condition update",
				"parentSecret", parentName)
			return nil // Not an error - parent may not exist yet
		}
		if err != nil {
			return fmt.Errorf("failed to get parent Secret: %w", err)
		}

		// Get existing conditions
		existingConditions := cm.getExistingConditions(secret)

		// Merge conditions (preserve lastTransitionTime if status/reason unchanged)
		mergedConditions := cm.mergeConditions(existingConditions, conditions)

		// Update annotations with conditions
		if secret.Annotations == nil {
			secret.Annotations = make(map[string]string)
		}

		// Store conditions as JSON in annotations
		for _, condition := range mergedConditions {
			conditionJSON, err := json.Marshal(condition)
			if err != nil {
				return fmt.Errorf("failed to marshal condition: %w", err)
			}
			annotationKey := fmt.Sprintf("status.conditions.%s", strings.ToLower(condition.Type))
			secret.Annotations[annotationKey] = string(conditionJSON)
		}

		// Update Secret
		_, err = cm.kubeClient.CoreV1().Secrets(namespace).Update(ctx, secret, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update parent Secret: %w", err)
		}

		logger.V(4).Info("Updated conditions on parent Secret",
			"parentSecret", parentName,
			"conditionCount", len(mergedConditions))
		return nil
	})
}

// getExistingConditions retrieves existing conditions from Secret annotations
func (cm *ConditionManager) getExistingConditions(secret *v1.Secret) []metav1.Condition {
	conditions := make([]metav1.Condition, 0, 4)

	conditionTypes := []string{"Ready", "Progressing", "Degraded", "Available"}
	for _, condType := range conditionTypes {
		annotationKey := fmt.Sprintf("status.conditions.%s", strings.ToLower(condType))
		if conditionJSON, ok := secret.Annotations[annotationKey]; ok {
			var condition metav1.Condition
			if err := json.Unmarshal([]byte(conditionJSON), &condition); err == nil {
				conditions = append(conditions, condition)
			}
		}
	}

	return conditions
}

// mergeConditions merges existing and new conditions, preserving lastTransitionTime when appropriate
func (cm *ConditionManager) mergeConditions(existing, new []metav1.Condition) []metav1.Condition {
	merged := make([]metav1.Condition, 0, len(new))
	existingMap := make(map[string]metav1.Condition)
	for _, cond := range existing {
		existingMap[cond.Type] = cond
	}

	for _, newCond := range new {
		existingCond, exists := existingMap[newCond.Type]
		if exists {
			// Preserve lastTransitionTime if status and reason haven't changed
			if existingCond.Status == newCond.Status && existingCond.Reason == newCond.Reason {
				newCond.LastTransitionTime = existingCond.LastTransitionTime
			}
		}
		merged = append(merged, newCond)
	}

	return merged
}

// GetConditions retrieves conditions from ApplySet parent Secret
func (cm *ConditionManager) GetConditions(
	ctx context.Context,
	releaseName, namespace string,
) ([]metav1.Condition, error) {
	parentName := fmt.Sprintf("%s%s", parent.ParentSecretNamePrefix, releaseName)

	secret, err := cm.kubeClient.CoreV1().Secrets(namespace).Get(ctx, parentName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return nil, fmt.Errorf("parent Secret not found: %s/%s", namespace, parentName)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get parent Secret: %w", err)
	}

	return cm.getExistingConditions(secret), nil
}
