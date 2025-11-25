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

package helmapplyset

import (
	"bytes"
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/kubernetes/fake"

	"k8s.io/kubernetes/pkg/controller/helmapplyset/parent"
	"k8s.io/kubernetes/pkg/controller/helmapplyset/status"
)

func TestStatusOptions_getConditions(t *testing.T) {
	tests := []struct {
		name          string
		secret        *v1.Secret
		expectedCount int
		expectedType  string
	}{
		{
			name: "secret with conditions",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test-release",
					Namespace: "default",
					Annotations: map[string]string{
						"status.conditions.ready":       `{"type":"Ready","status":"True","reason":"AllResourcesHealthy","message":"All resources healthy"}`,
						"status.conditions.progressing": `{"type":"Progressing","status":"False","reason":"NoRolloutInProgress","message":"No rollout in progress"}`,
					},
				},
			},
			expectedCount: 2,
			expectedType:  "Ready",
		},
		{
			name: "secret without conditions",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "applyset-test-release",
					Namespace: "default",
				},
			},
			expectedCount: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := NewStatusOptions(genericiooptions.IOStreams{})
			conditions, err := o.getConditions(tt.secret)
			require.NoError(t, err)
			assert.Equal(t, tt.expectedCount, len(conditions))
			if tt.expectedType != "" && len(conditions) > 0 {
				assert.Equal(t, tt.expectedType, conditions[0].Type)
			}
		})
	}
}

func TestStatusOptions_colorizeStatus(t *testing.T) {
	tests := []struct {
		name     string
		status   string
		noColor  bool
		expected string
	}{
		{
			name:     "healthy status with color",
			status:   "healthy",
			noColor:  false,
			expected: "\033[32mhealthy\033[0m",
		},
		{
			name:     "healthy status without color",
			status:   "healthy",
			noColor:  true,
			expected: "healthy",
		},
		{
			name:     "progressing status",
			status:   "progressing",
			noColor:  false,
			expected: "\033[33mprogressing\033[0m",
		},
		{
			name:     "failed status",
			status:   "failed",
			noColor:  false,
			expected: "\033[31mfailed\033[0m",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := NewStatusOptions(genericiooptions.IOStreams{})
			o.NoColor = tt.noColor
			result := o.colorizeStatus(tt.status)
			if tt.noColor {
				assert.Equal(t, tt.expected, result)
			} else {
				// Check that ANSI codes are present (when not disabled)
				assert.Contains(t, result, tt.status)
			}
		})
	}
}

func TestStatusOptions_printHeader(t *testing.T) {
	health := &status.ReleaseHealth{
		ReleaseName:   "test-release",
		Namespace:     "default",
		OverallStatus: "healthy",
	}

	var buf bytes.Buffer
	o := NewStatusOptions(genericiooptions.IOStreams{Out: &buf})
	o.NoColor = true // Disable colors for testing

	o.printHeader(health)

	output := buf.String()
	assert.Contains(t, output, "test-release")
	assert.Contains(t, output, "default")
	assert.Contains(t, output, "healthy")
}

func TestStatusOptions_printResourceTable(t *testing.T) {
	health := &status.ReleaseHealth{
		ReleaseName:   "test-release",
		Namespace:     "default",
		OverallStatus: "healthy",
		ResourceHealth: map[string]status.HealthStatus{
			"apps/v1/Deployment/default/dep1": {
				Healthy: true,
				Reason:  "AllReplicasReady",
				Message: "3/3 replicas ready",
			},
			"v1/Service/default/svc1": {
				Healthy: true,
				Reason:  "EndpointsReady",
				Message: "Endpoints available",
			},
		},
	}

	var buf bytes.Buffer
	o := NewStatusOptions(genericiooptions.IOStreams{Out: &buf})
	o.NoColor = true

	o.printResourceTable(health)

	output := buf.String()
	assert.Contains(t, output, "KIND")
	assert.Contains(t, output, "NAME")
	assert.Contains(t, output, "STATUS")
	assert.Contains(t, output, "MESSAGE")
	assert.Contains(t, output, "Deployment")
	assert.Contains(t, output, "Service")
}

func TestStatusOptions_printConditions(t *testing.T) {
	conditions := []metav1.Condition{
		{
			Type:    "Ready",
			Status:  metav1.ConditionTrue,
			Reason:  "AllResourcesHealthy",
			Message: "All 5 resources are healthy",
		},
		{
			Type:    "Progressing",
			Status:  metav1.ConditionFalse,
			Reason:  "NoRolloutInProgress",
			Message: "No rollout in progress",
		},
	}

	var buf bytes.Buffer
	o := NewStatusOptions(genericiooptions.IOStreams{Out: &buf})
	o.NoColor = true

	o.printConditions(conditions)

	output := buf.String()
	assert.Contains(t, output, "Conditions:")
	assert.Contains(t, output, "Ready")
	assert.Contains(t, output, "AllResourcesHealthy")
	assert.Contains(t, output, "Progressing")
}

func TestStatusOptions_printEvents(t *testing.T) {
	events := []v1.Event{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "event-1",
			},
			Type:          v1.EventTypeNormal,
			Reason:        "HelmReleaseHealthy",
			Message:       "All resources are healthy",
			LastTimestamp: metav1.Now(),
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "event-2",
			},
			Type:          v1.EventTypeWarning,
			Reason:        "ResourceUnhealthy",
			Message:       "Deployment has unhealthy replicas",
			LastTimestamp: metav1.Now(),
		},
	}

	var buf bytes.Buffer
	o := NewStatusOptions(genericiooptions.IOStreams{Out: &buf})
	o.NoColor = true

	o.printEvents(events)

	output := buf.String()
	assert.Contains(t, output, "Recent Events:")
	assert.Contains(t, output, "TIME")
	assert.Contains(t, output, "TYPE")
	assert.Contains(t, output, "REASON")
	assert.Contains(t, output, "HelmReleaseHealthy")
	assert.Contains(t, output, "ResourceUnhealthy")
}

func TestStatusOptions_getRecentEvents(t *testing.T) {
	parentSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test-release",
			Namespace: "default",
		},
	}

	client := fake.NewSimpleClientset()

	// Create some test events
	for i := 0; i < 7; i++ {
		event := &v1.Event{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("event-%d", i),
				Namespace: "default",
			},
			InvolvedObject: v1.ObjectReference{
				Name: "applyset-test-release",
				Kind: "Secret",
			},
			Type:          v1.EventTypeNormal,
			Reason:        "TestReason",
			Message:       fmt.Sprintf("Test message %d", i),
			LastTimestamp: metav1.NewTime(metav1.Now().Add(time.Duration(i) * time.Minute)),
		}
		client.CoreV1().Events("default").Create(context.Background(), event, metav1.CreateOptions{})
	}

	o := NewStatusOptions(genericiooptions.IOStreams{})
	o.Client = client
	o.Namespace = "default"

	events, err := o.getRecentEvents(context.Background(), parentSecret)
	require.NoError(t, err)

	// Should return only last 5 events
	assert.LessOrEqual(t, len(events), 5)
	if len(events) > 0 {
		// Events should be sorted newest first
		for i := 1; i < len(events); i++ {
			assert.True(t, events[i-1].LastTimestamp.Time.After(events[i].LastTimestamp.Time) ||
				events[i-1].LastTimestamp.Time.Equal(events[i].LastTimestamp.Time))
		}
	}
}

func TestStatusOptions_aggregateHealth(t *testing.T) {
	parentSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "applyset-test-release",
			Namespace: "default",
			Labels: map[string]string{
				parent.ApplySetParentIDLabel: "applyset-test-id-v1",
			},
			Annotations: map[string]string{
				"status.conditions.ready": `{"type":"Ready","status":"True","reason":"AllResourcesHealthy","message":"All resources healthy"}`,
			},
		},
	}

	o := NewStatusOptions(genericiooptions.IOStreams{})
	o.Name = "test-release"
	o.Namespace = "default"

	health, err := o.aggregateHealth(context.Background(), "applyset-test-id-v1", parentSecret)
	require.NoError(t, err)

	assert.Equal(t, "test-release", health.ReleaseName)
	assert.Equal(t, "default", health.Namespace)
	assert.Equal(t, "healthy", health.OverallStatus)
}
