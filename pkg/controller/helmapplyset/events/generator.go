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

package events

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/status"
)

const (
	// EventReasonReleaseHealthy indicates all resources are healthy
	EventReasonReleaseHealthy = "HelmReleaseHealthy"

	// EventReasonReleaseDegraded indicates some resources are unhealthy
	EventReasonReleaseDegraded = "HelmReleaseDegraded"

	// EventReasonReleaseProgressing indicates rollout is in progress
	EventReasonReleaseProgressing = "HelmReleaseProgressing"

	// EventReasonReleaseFailed indicates critical failure
	EventReasonReleaseFailed = "HelmReleaseFailed"

	// EventReasonResourceUnhealthy indicates a specific resource failed
	EventReasonResourceUnhealthy = "ResourceUnhealthy"

	// EventReasonResourceRecovered indicates a resource recovered from failure
	EventReasonResourceRecovered = "ResourceRecovered"

	// EventDeduplicationWindow is the time window for event deduplication
	EventDeduplicationWindow = 5 * time.Minute
)

// EventGenerator generates Kubernetes Events for status changes
type EventGenerator struct {
	recorder record.EventRecorder
	logger   klog.Logger

	// Track last emitted events for deduplication
	lastEvents map[string]lastEvent
	mu         sync.RWMutex
}

// lastEvent tracks the last emitted event for deduplication
type lastEvent struct {
	reason    string
	message   string
	timestamp time.Time
}

// NewEventGenerator creates a new event generator
func NewEventGenerator(
	recorder record.EventRecorder,
	logger klog.Logger,
) *EventGenerator {
	return &EventGenerator{
		recorder:   recorder,
		logger:     logger,
		lastEvents: make(map[string]lastEvent),
	}
}

// EmitReleaseStatusEvent emits an event for release status changes
func (eg *EventGenerator) EmitReleaseStatusEvent(
	ctx context.Context,
	releaseName, namespace string,
	health *status.ReleaseHealth,
	parentSecret *v1.Secret,
) {
	logger := klog.FromContext(ctx)
	eventKey := fmt.Sprintf("%s/%s", namespace, releaseName)

	// Determine event reason and message based on status
	var reason string
	var message string

	switch health.OverallStatus {
	case "healthy":
		reason = EventReasonReleaseHealthy
		message = eg.formatHealthyMessage(releaseName, health)

	case "progressing":
		reason = EventReasonReleaseProgressing
		message = eg.formatProgressingMessage(releaseName, health)

	case "degraded":
		reason = EventReasonReleaseDegraded
		message = eg.formatDegradedMessage(releaseName, health)

	case "failed":
		reason = EventReasonReleaseFailed
		message = eg.formatFailedMessage(releaseName, health)

	default:
		// Don't emit events for unknown status
		return
	}

	// Check if we should emit this event (deduplication)
	if !eg.shouldEmitEvent(eventKey, reason, message) {
		logger.V(4).Info("Skipping duplicate event",
			"release", releaseName,
			"reason", reason)
		return
	}

	// Emit event
	eg.recorder.Eventf(parentSecret, v1.EventTypeNormal, reason, message)

	// Record event for deduplication
	eg.recordEvent(eventKey, reason, message)

	logger.Info("Emitted release status event",
		"release", releaseName,
		"namespace", namespace,
		"reason", reason,
		"status", health.OverallStatus)
}

// EmitResourceEvent emits an event for individual resource status changes
func (eg *EventGenerator) EmitResourceEvent(
	ctx context.Context,
	releaseName, namespace string,
	resourceKey string,
	healthStatus status.HealthStatus,
	parentSecret *v1.Secret,
) {
	logger := klog.FromContext(ctx)
	eventKey := fmt.Sprintf("%s/%s/%s", namespace, releaseName, resourceKey)

	var reason string
	var eventType string
	var message string

	if healthStatus.Healthy {
		reason = EventReasonResourceRecovered
		eventType = v1.EventTypeNormal
		message = eg.formatResourceRecoveredMessage(releaseName, resourceKey, healthStatus)
	} else {
		reason = EventReasonResourceUnhealthy
		eventType = v1.EventTypeWarning
		message = eg.formatResourceUnhealthyMessage(releaseName, resourceKey, healthStatus)
	}

	// Check if we should emit this event (deduplication)
	if !eg.shouldEmitEvent(eventKey, reason, message) {
		logger.V(4).Info("Skipping duplicate resource event",
			"release", releaseName,
			"resource", resourceKey,
			"reason", reason)
		return
	}

	// Emit event
	eg.recorder.Eventf(parentSecret, eventType, reason, message)

	// Record event for deduplication
	eg.recordEvent(eventKey, reason, message)

	logger.V(4).Info("Emitted resource event",
		"release", releaseName,
		"resource", resourceKey,
		"reason", reason)
}

// formatHealthyMessage formats message for healthy release
func (eg *EventGenerator) formatHealthyMessage(releaseName string, health *status.ReleaseHealth) string {
	return fmt.Sprintf("Helm release '%s' is healthy. All %d resources are ready.",
		releaseName, health.TotalResources)
}

// formatProgressingMessage formats message for progressing release
func (eg *EventGenerator) formatProgressingMessage(releaseName string, health *status.ReleaseHealth) string {
	return fmt.Sprintf("Helm release '%s' rollout in progress. %d resources are progressing: %s. %d/%d resources are healthy.",
		releaseName, health.ProgressingResources, strings.Join(health.GetProgressingResourceNames(), ", "),
		health.HealthyResources, health.TotalResources)
}

// formatDegradedMessage formats message for degraded release
func (eg *EventGenerator) formatDegradedMessage(releaseName string, health *status.ReleaseHealth) string {
	return fmt.Sprintf("Helm release '%s' is degraded. %d resources are unhealthy: %s. %d/%d resources are healthy.",
		releaseName, health.DegradedResources, strings.Join(health.GetUnhealthyResourceNames(), ", "),
		health.HealthyResources, health.TotalResources)
}

// formatFailedMessage formats message for failed release
func (eg *EventGenerator) formatFailedMessage(releaseName string, health *status.ReleaseHealth) string {
	return fmt.Sprintf("Helm release '%s' has failed. %d resources failed: %s.",
		releaseName, health.FailedResources, strings.Join(health.GetFailedResourceNames(), ", "))
}

// formatResourceUnhealthyMessage formats message for unhealthy resource
func (eg *EventGenerator) formatResourceUnhealthyMessage(releaseName, resourceKey string, healthStatus status.HealthStatus) string {
	// Extract resource name from key (format: "gvk/namespace/name")
	parts := strings.Split(resourceKey, "/")
	resourceName := resourceKey
	if len(parts) >= 3 {
		resourceName = fmt.Sprintf("%s/%s", parts[len(parts)-2], parts[len(parts)-1])
	}

	return fmt.Sprintf("Resource %s in release '%s' is unhealthy: %s. %s",
		resourceName, releaseName, healthStatus.Reason, healthStatus.Message)
}

// formatResourceRecoveredMessage formats message for recovered resource
func (eg *EventGenerator) formatResourceRecoveredMessage(releaseName, resourceKey string, healthStatus status.HealthStatus) string {
	parts := strings.Split(resourceKey, "/")
	resourceName := resourceKey
	if len(parts) >= 3 {
		resourceName = fmt.Sprintf("%s/%s", parts[len(parts)-2], parts[len(parts)-1])
	}

	return fmt.Sprintf("Resource %s in release '%s' recovered: %s",
		resourceName, releaseName, healthStatus.Message)
}

// shouldEmitEvent checks if an event should be emitted (deduplication)
func (eg *EventGenerator) shouldEmitEvent(eventKey, reason, message string) bool {
	eg.mu.RLock()
	defer eg.mu.RUnlock()

	lastEvent, exists := eg.lastEvents[eventKey]
	if !exists {
		return true
	}

	// Emit if reason or message changed
	if lastEvent.reason != reason || lastEvent.message != message {
		return true
	}

	// Emit if enough time has passed since last event
	if time.Since(lastEvent.timestamp) > EventDeduplicationWindow {
		return true
	}

	return false
}

// recordEvent records an event for deduplication
func (eg *EventGenerator) recordEvent(eventKey, reason, message string) {
	eg.mu.Lock()
	defer eg.mu.Unlock()

	eg.lastEvents[eventKey] = lastEvent{
		reason:    reason,
		message:   message,
		timestamp: time.Now(),
	}

	// Clean up old events (keep map size reasonable)
	if len(eg.lastEvents) > 1000 {
		eg.cleanupOldEvents()
	}
}

// cleanupOldEvents removes events older than deduplication window
func (eg *EventGenerator) cleanupOldEvents() {
	cutoff := time.Now().Add(-EventDeduplicationWindow)
	for key, event := range eg.lastEvents {
		if event.timestamp.Before(cutoff) {
			delete(eg.lastEvents, key)
		}
	}
}
