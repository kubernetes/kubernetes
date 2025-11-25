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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2/ktesting"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/status"
)

func TestEventGenerator_EmitReleaseStatusEvent(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	parentSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test-release",
			Namespace: "default",
		},
	}

	tests := []struct {
		name           string
		health         *status.ReleaseHealth
		expectedReason string
		expectedType   string
	}{
		{
			name: "healthy release",
			health: &status.ReleaseHealth{
				ReleaseName:      "test-release",
				Namespace:        "default",
				OverallStatus:    "healthy",
				TotalResources:   5,
				HealthyResources: 5,
			},
			expectedReason: EventReasonReleaseHealthy,
			expectedType:   v1.EventTypeNormal,
		},
		{
			name: "degraded release",
			health: &status.ReleaseHealth{
				ReleaseName:       "test-release",
				Namespace:         "default",
				OverallStatus:     "degraded",
				TotalResources:    5,
				HealthyResources:  3,
				DegradedResources: 2,
				ResourceHealth: map[string]status.HealthStatus{
					"apps/v1/Deployment/default/dep1": {Healthy: false, Reason: "ReplicasNotReady"},
				},
			},
			expectedReason: EventReasonReleaseDegraded,
			expectedType:   v1.EventTypeNormal,
		},
		{
			name: "progressing release",
			health: &status.ReleaseHealth{
				ReleaseName:          "test-release",
				Namespace:            "default",
				OverallStatus:        "progressing",
				TotalResources:       5,
				HealthyResources:     3,
				ProgressingResources: 2,
				ResourceHealth: map[string]status.HealthStatus{
					"apps/v1/Deployment/default/dep1": {Healthy: false, Reason: "Progressing"},
				},
			},
			expectedReason: EventReasonReleaseProgressing,
			expectedType:   v1.EventTypeNormal,
		},
		{
			name: "failed release",
			health: &status.ReleaseHealth{
				ReleaseName:      "test-release",
				Namespace:        "default",
				OverallStatus:    "failed",
				TotalResources:   5,
				HealthyResources: 3,
				FailedResources:  2,
				ResourceHealth: map[string]status.HealthStatus{
					"batch/v1/Job/default/job1": {Healthy: false, Reason: "Failed"},
				},
			},
			expectedReason: EventReasonReleaseFailed,
			expectedType:   v1.EventTypeNormal,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clear recorder
			recorder := record.NewFakeRecorder(100)
			generator := NewEventGenerator(recorder, logger)

			generator.EmitReleaseStatusEvent(ctx, tt.health.ReleaseName, tt.health.Namespace, tt.health, parentSecret)

			// Check event was emitted
			select {
			case event := <-recorder.Events:
				assert.Contains(t, event, tt.expectedReason)
				assert.Contains(t, event, tt.expectedType)
			case <-time.After(1 * time.Second):
				t.Fatal("Expected event but none was emitted")
			}
		})
	}
}

func TestEventGenerator_EventDeduplication(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	recorder := record.NewFakeRecorder(100)
	generator := NewEventGenerator(recorder, logger)

	parentSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test-release",
			Namespace: "default",
		},
	}

	health := &status.ReleaseHealth{
		ReleaseName:      "test-release",
		Namespace:        "default",
		OverallStatus:    "healthy",
		TotalResources:   5,
		HealthyResources: 5,
	}

	// Emit first event
	generator.EmitReleaseStatusEvent(ctx, health.ReleaseName, health.Namespace, health, parentSecret)

	// Wait for event
	select {
	case <-recorder.Events:
		// Good
	case <-time.After(1 * time.Second):
		t.Fatal("Expected first event")
	}

	// Emit same event again immediately (should be deduplicated)
	generator.EmitReleaseStatusEvent(ctx, health.ReleaseName, health.Namespace, health, parentSecret)

	// Should not emit duplicate event
	select {
	case <-recorder.Events:
		t.Error("Duplicate event should not be emitted")
	case <-time.After(100 * time.Millisecond):
		// Good - no duplicate event
	}

	// Change status and emit again (should emit)
	health.OverallStatus = "degraded"
	health.DegradedResources = 2
	generator.EmitReleaseStatusEvent(ctx, health.ReleaseName, health.Namespace, health, parentSecret)

	// Should emit new event
	select {
	case event := <-recorder.Events:
		assert.Contains(t, event, EventReasonReleaseDegraded)
	case <-time.After(1 * time.Second):
		t.Fatal("Expected new event for status change")
	}
}

func TestEventGenerator_EmitResourceEvent(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)

	parentSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test-release",
			Namespace: "default",
		},
	}

	tests := []struct {
		name           string
		healthStatus   status.HealthStatus
		expectedReason string
		expectedType   string
	}{
		{
			name: "unhealthy resource",
			healthStatus: status.HealthStatus{
				Healthy: false,
				Reason:  "ReplicasNotReady",
				Message: "Deployment has 1/3 ready replicas",
			},
			expectedReason: EventReasonResourceUnhealthy,
			expectedType:   v1.EventTypeWarning,
		},
		{
			name: "recovered resource",
			healthStatus: status.HealthStatus{
				Healthy: true,
				Reason:  "AllReplicasReady",
				Message: "Deployment has 3/3 ready replicas",
			},
			expectedReason: EventReasonResourceRecovered,
			expectedType:   v1.EventTypeNormal,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clear recorder
			recorder := record.NewFakeRecorder(100)
			generator := NewEventGenerator(recorder, logger)

			generator.EmitResourceEvent(ctx, "test-release", "default", "apps/v1/Deployment/default/dep1", tt.healthStatus, parentSecret)

			select {
			case event := <-recorder.Events:
				assert.Contains(t, event, tt.expectedReason)
				assert.Contains(t, event, tt.expectedType)
			case <-time.After(1 * time.Second):
				t.Fatal("Expected event but none was emitted")
			}
		})
	}
}

func TestEventGenerator_formatMessages(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	recorder := record.NewFakeRecorder(100)
	generator := NewEventGenerator(recorder, logger)

	tests := []struct {
		name     string
		health   *status.ReleaseHealth
		formatFn func(*EventGenerator, string, *status.ReleaseHealth) string
		checkFn  func(*testing.T, string)
	}{
		{
			name: "healthy message",
			health: &status.ReleaseHealth{
				ReleaseName:      "test-release",
				TotalResources:   5,
				HealthyResources: 5,
			},
			formatFn: func(eg *EventGenerator, name string, h *status.ReleaseHealth) string {
				return eg.formatHealthyMessage(name, h)
			},
			checkFn: func(t *testing.T, msg string) {
				assert.Contains(t, msg, "test-release")
				assert.Contains(t, msg, "healthy")
				assert.Contains(t, msg, "5")
			},
		},
		{
			name: "degraded message",
			health: &status.ReleaseHealth{
				ReleaseName:       "test-release",
				TotalResources:    5,
				HealthyResources:  3,
				DegradedResources: 2,
				ResourceHealth: map[string]status.HealthStatus{
					"apps/v1/Deployment/default/dep1": {Healthy: false, Reason: "ReplicasNotReady"},
				},
			},
			formatFn: func(eg *EventGenerator, name string, h *status.ReleaseHealth) string {
				return eg.formatDegradedMessage(name, h)
			},
			checkFn: func(t *testing.T, msg string) {
				assert.Contains(t, msg, "test-release")
				assert.Contains(t, msg, "degraded")
				assert.Contains(t, msg, "2")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := tt.formatFn(generator, tt.health.ReleaseName, tt.health)
			tt.checkFn(t, msg)
		})
	}
}
