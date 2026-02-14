/*
Copyright 2025 The Kubernetes Authors.

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

package lifecycle

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
)

func TestPrinterJSON(t *testing.T) {
	var buf bytes.Buffer
	streams := genericiooptions.IOStreams{Out: &buf, ErrOut: &buf}
	printer := NewPrinter(streams, "json", false)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodPending},
	}

	result := &DiagnosticResult{
		Pod:   pod,
		Phase: PhasePending,
		RootCause: RootCause{
			Category:   "Scheduling",
			Summary:    "Test summary",
			Confidence: 0.95,
		},
	}

	err := printer.Print(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Verify it's valid JSON
	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(output), &parsed); err != nil {
		t.Errorf("output is not valid JSON: %v", err)
	}

	// Verify key fields are present
	if parsed["phase"] != "Pending" {
		t.Errorf("expected phase 'Pending', got %v", parsed["phase"])
	}
}

func TestPrinterYAML(t *testing.T) {
	var buf bytes.Buffer
	streams := genericiooptions.IOStreams{Out: &buf, ErrOut: &buf}
	printer := NewPrinter(streams, "yaml", false)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodPending},
	}

	result := &DiagnosticResult{
		Pod:   pod,
		Phase: PhasePending,
		RootCause: RootCause{
			Category:   "Scheduling",
			Summary:    "Test summary",
			Confidence: 0.95,
		},
	}

	err := printer.Print(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Verify YAML format indicators
	if !strings.Contains(output, "phase: Pending") {
		t.Error("expected YAML output to contain 'phase: Pending'")
	}
}

func TestPrinterHuman(t *testing.T) {
	var buf bytes.Buffer
	streams := genericiooptions.IOStreams{Out: &buf, ErrOut: &buf}
	printer := NewPrinter(streams, "", false) // empty format = human

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: "node-1"},
		Status:     corev1.PodStatus{Phase: corev1.PodPending},
	}

	result := &DiagnosticResult{
		Pod:   pod,
		Phase: PhaseSchedulingFailed,
		RootCause: RootCause{
			Category:   "Scheduling",
			Summary:    "Pod cannot be scheduled",
			Details:    "Detailed explanation here",
			Confidence: 0.95,
		},
		SchedulingInfo: &SchedulingInfo{
			Message:        "0/3 nodes are available",
			AvailableNodes: "0",
			TotalNodes:     "3",
			FailureReasons: []string{"2 Insufficient cpu"},
		},
		Recommendations: []Recommendation{
			{Priority: 1, Action: "Test action", Command: "kubectl get pods"},
		},
	}

	err := printer.Print(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Verify human-readable output structure
	expectedStrings := []string{
		"Pod Lifecycle Diagnostic Report",
		"default/test-pod",
		"Root Cause Analysis",
		"Scheduling",
		"Recommendations",
	}

	for _, expected := range expectedStrings {
		if !strings.Contains(output, expected) {
			t.Errorf("expected output to contain %q", expected)
		}
	}
}

func TestPrinterHumanShowDetails(t *testing.T) {
	var buf bytes.Buffer
	streams := genericiooptions.IOStreams{Out: &buf, ErrOut: &buf}
	printer := NewPrinter(streams, "", true) // show-details mode

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodPending},
	}

	result := &DiagnosticResult{
		Pod:   pod,
		Phase: PhaseCrashLooping,
		RootCause: RootCause{
			Category:   "Application",
			Summary:    "Application error",
			Details:    "This is the detailed explanation",
			Confidence: 0.7,
		},
		ContainerIssues: []ContainerIssue{
			{
				ContainerName: "main",
				State:         "CrashLoopBackOff",
				ExitCode:      1,
				RestartCount:  5,
				LastLogs:      "Error: connection refused\nFailed to start",
			},
		},
		Recommendations: []Recommendation{
			{Priority: 1, Action: "Check logs", Command: "kubectl logs test-pod --previous", Explanation: "Logs show the error"},
			{Priority: 2, Action: "Second action"},
			{Priority: 3, Action: "Third action"},
			{Priority: 4, Action: "Fourth action"}, // Should be shown with --show-details
		},
		Events: []EventSummary{
			{Type: "Warning", Reason: "Failed", Message: "Container failed"},
		},
	}

	err := printer.Print(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// With --show-details, should include details
	if !strings.Contains(output, "This is the detailed explanation") {
		t.Error("--show-details should include details")
	}

	// Should include previous logs
	if !strings.Contains(output, "Error: connection refused") {
		t.Error("--show-details should include previous logs")
	}

	// Should include all recommendations (not just first 3)
	if !strings.Contains(output, "Fourth action") {
		t.Error("--show-details should include all recommendations")
	}

	// Should include events
	if !strings.Contains(output, "Recent Events") {
		t.Error("--show-details should include events section")
	}
}

func TestPrinterHumanNoNode(t *testing.T) {
	var buf bytes.Buffer
	streams := genericiooptions.IOStreams{Out: &buf, ErrOut: &buf}
	printer := NewPrinter(streams, "", false)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Spec:       corev1.PodSpec{NodeName: ""}, // Not scheduled
		Status:     corev1.PodStatus{Phase: corev1.PodPending},
	}

	result := &DiagnosticResult{
		Pod:   pod,
		Phase: PhaseSchedulingFailed,
		RootCause: RootCause{
			Category: "Scheduling",
			Summary:  "Pod cannot be scheduled",
		},
	}

	err := printer.Print(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	if !strings.Contains(output, "<not scheduled>") {
		t.Error("expected '<not scheduled>' for unscheduled pod")
	}
}

func TestTranslateAge(t *testing.T) {
	tests := []struct {
		name     string
		time     time.Time
		expected string
	}{
		{
			name:     "zero time",
			time:     time.Time{},
			expected: "<unknown>",
		},
		{
			name:     "30 seconds ago",
			time:     time.Now().Add(-30 * time.Second),
			expected: "30s",
		},
		{
			name:     "5 minutes ago",
			time:     time.Now().Add(-5 * time.Minute),
			expected: "5m",
		},
		{
			name:     "2 hours ago",
			time:     time.Now().Add(-2 * time.Hour),
			expected: "2h",
		},
		{
			name:     "3 days ago",
			time:     time.Now().Add(-3 * 24 * time.Hour),
			expected: "3d",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := translateAge(tc.time)
			if result != tc.expected {
				t.Errorf("expected %q, got %q", tc.expected, result)
			}
		})
	}
}

func TestPrinterContainerIssues(t *testing.T) {
	var buf bytes.Buffer
	streams := genericiooptions.IOStreams{Out: &buf, ErrOut: &buf}
	printer := NewPrinter(streams, "", false)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodRunning},
	}

	result := &DiagnosticResult{
		Pod:   pod,
		Phase: PhaseCrashLooping,
		RootCause: RootCause{
			Category: "Application",
			Summary:  "Container crashing",
		},
		ContainerIssues: []ContainerIssue{
			{
				ContainerName: "main",
				State:         "CrashLoopBackOff",
				Message:       "Back-off restarting container",
				ExitCode:      137,
				RestartCount:  10,
			},
		},
	}

	err := printer.Print(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	expectedStrings := []string{
		"Container Issues",
		"main",
		"CrashLoopBackOff",
		"137", // exit code
		"10",  // restart count
	}

	for _, expected := range expectedStrings {
		if !strings.Contains(output, expected) {
			t.Errorf("expected output to contain %q", expected)
		}
	}
}

func TestPrinterVolumeIssues(t *testing.T) {
	var buf bytes.Buffer
	streams := genericiooptions.IOStreams{Out: &buf, ErrOut: &buf}
	printer := NewPrinter(streams, "", false)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodPending},
	}

	result := &DiagnosticResult{
		Pod:   pod,
		Phase: PhaseContainerCreating,
		RootCause: RootCause{
			Category: "Volume",
			Summary:  "PVC pending",
		},
		VolumeIssues: []VolumeIssue{
			{
				VolumeName: "data-volume",
				Type:       "PVCPending",
				Message:    "Waiting for volume binding",
			},
		},
	}

	err := printer.Print(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	expectedStrings := []string{
		"Volume Issues",
		"data-volume",
		"PVCPending",
		"Waiting for volume binding",
	}

	for _, expected := range expectedStrings {
		if !strings.Contains(output, expected) {
			t.Errorf("expected output to contain %q", expected)
		}
	}
}

func TestPrinterRecommendationsLimit(t *testing.T) {
	var buf bytes.Buffer
	streams := genericiooptions.IOStreams{Out: &buf, ErrOut: &buf}
	printer := NewPrinter(streams, "", false) // without --show-details

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "default"},
		Status:     corev1.PodStatus{Phase: corev1.PodPending},
	}

	result := &DiagnosticResult{
		Pod:   pod,
		Phase: PhasePending,
		RootCause: RootCause{
			Category: "Unknown",
			Summary:  "Unknown issue",
		},
		Recommendations: []Recommendation{
			{Priority: 1, Action: "First action"},
			{Priority: 2, Action: "Second action"},
			{Priority: 3, Action: "Third action"},
			{Priority: 4, Action: "Fourth action"},
			{Priority: 5, Action: "Fifth action"},
		},
	}

	err := printer.Print(result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	output := buf.String()

	// Without --show-details, only the first 3 recommendations should be shown
	if !strings.Contains(output, "First action") {
		t.Error("should show first action")
	}
	if !strings.Contains(output, "Second action") {
		t.Error("should show second action")
	}
	if !strings.Contains(output, "Third action") {
		t.Error("should show third action")
	}
	if strings.Contains(output, "Fourth action") {
		t.Error("should NOT show fourth action without --show-details")
	}
	if !strings.Contains(output, "Use --show-details for more recommendations") {
		t.Error("should show hint to use --show-details")
	}
}
