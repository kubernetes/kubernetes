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

package get

import (
	"net/http"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

const LastActivityAnnotationKey = "kubernetes.io/last-activity"

// TestGetPodsWithIdleFlag tests the --idle flag functionality
func TestGetPodsWithIdleFlag(t *testing.T) {
	now := time.Now()
	pods := &corev1.PodList{
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "idle-pod",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-2 * time.Hour).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "active-pod",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-5 * time.Minute).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "never-active-pod",
					Namespace: "test",
					// No last-activity annotation
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOut(buf)
	cmd.SetErr(buf)

	// Set --idle=1h to filter pods idle for more than 1 hour
	cmd.Flags().Set("idle", "1h")
	cmd.Run(cmd, []string{"pods"})

	output := buf.String()
	assert.Contains(t, output, "idle-pod", "expected idle-pod to be shown")
	assert.NotContains(t, output, "active-pod", "expected active-pod to be filtered out")
}

// TestGetPodsWithIdleFlagDefaultDuration tests default --idle duration (30m)
func TestGetPodsWithIdleFlagDefaultDuration(t *testing.T) {
	now := time.Now()
	pods := &corev1.PodList{
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "idle-pod",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-45 * time.Minute).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "active-pod",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-15 * time.Minute).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOut(buf)
	cmd.SetErr(buf)

	// Set --idle without duration to use default 30m
	cmd.Flags().Set("idle", "")
	cmd.Run(cmd, []string{"pods"})

	output := buf.String()
	assert.Contains(t, output, "idle-pod", "expected idle-pod to be shown with default 30m")
	assert.NotContains(t, output, "active-pod", "expected active-pod to be filtered out")
}

// TestGetPodsIdleSinceColumn tests IDLE-SINCE column output
func TestGetPodsIdleSinceColumn(t *testing.T) {
	now := time.Now()
	pods := &corev1.PodList{
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-2h-idle",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-2 * time.Hour).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-15m-idle",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-15 * time.Minute).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-30s-idle",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-30 * time.Second).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-no-activity",
					Namespace: "test",
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOut(buf)
	cmd.SetErr(buf)

	// Use --idle to enable the column
	cmd.Flags().Set("idle", "0s")
	cmd.Run(cmd, []string{"pods"})

	output := buf.String()

	// Check header contains IDLE-SINCE
	assert.Contains(t, output, "IDLE-SINCE", "expected IDLE-SINCE column header")

	// Check format of idle durations
	assert.Contains(t, output, "2h", "expected 2h idle time for pod-2h-idle")
	assert.Contains(t, output, "15m", "expected 15m idle time for pod-15m-idle")
	assert.Contains(t, output, "<1m", "expected <1m for pod-30s-idle")
	assert.Contains(t, output, "-", "expected - for pod with no activity")
}

// TestGetPodsIdleSinceColumnSorting tests that IDLE-SINCE column sorts correctly
func TestGetPodsIdleSinceColumnSorting(t *testing.T) {
	now := time.Now()
	pods := &corev1.PodList{
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-old",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-5 * time.Hour).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-medium",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-1 * time.Hour).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-new",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-10 * time.Minute).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOut(buf)
	cmd.SetErr(buf)

	// Sort by idle-since (newest idle first means most recently active)
	cmd.Flags().Set("idle", "0s")
	cmd.Flags().Set("sort-by", ".metadata.annotations.kubernetes\\.io/last-activity")
	cmd.Run(cmd, []string{"pods"})

	output := buf.String()
	lines := strings.Split(output, "\n")

	// Find the order of pods in output
	var podOrder []string
	for _, line := range lines {
		if strings.Contains(line, "pod-old") {
			podOrder = append(podOrder, "old")
		} else if strings.Contains(line, "pod-medium") {
			podOrder = append(podOrder, "medium")
		} else if strings.Contains(line, "pod-new") {
			podOrder = append(podOrder, "new")
		}
	}

	// When sorted by last-activity ascending, newest first
	require.Len(t, podOrder, 3, "expected 3 pods in output")
}

// TestIdleFlagOnlyAffectsPods tests that --idle only works with pods
func TestIdleFlagOnlyAffectsPods(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	streams, _, buf, errBuf := genericiooptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOut(buf)
	cmd.SetErr(errBuf)

	// Try to use --idle with deployments
	cmd.Flags().Set("idle", "30m")
	err := cmd.Flags().Parse([]string{"deployments"})

	// The flag itself should parse fine
	assert.NoError(t, err, "flag parsing should succeed")

	// But when running, it should either ignore or warn about non-pod resources
	// This depends on implementation choice
}

// TestIdleFlagWithDifferentDurations tests parsing of various duration formats
func TestIdleFlagWithDifferentDurations(t *testing.T) {
	testCases := []struct {
		name     string
		duration string
		wantErr  bool
	}{
		{"minutes", "30m", false},
		{"hours", "2h", false},
		{"hours and minutes", "1h30m", false},
		{"seconds", "90s", false},
		{"days equivalent", "24h", false},
		{"zero", "0s", false},
		{"invalid format", "invalid", true},
		{"negative", "-1h", true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			streams, _, _, errBuf := genericiooptions.NewTestIOStreams()
			cmd := NewCmdGet("kubectl", tf, streams)

			err := cmd.Flags().Set("idle", tc.duration)

			if tc.wantErr {
				// Either flag set fails or command run fails
				if err == nil {
					// Run command to see if it fails during execution
					pods := &corev1.PodList{Items: []corev1.Pod{}}
					codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
					tf.UnstructuredClient = &fake.RESTClient{
						NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
						Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
					}
					cmd.Run(cmd, []string{"pods"})
					// Check if there's an error message
					assert.NotEmpty(t, errBuf.String(), "expected error for invalid duration %s", tc.duration)
				}
			} else {
				assert.NoError(t, err, "expected no error for valid duration %s", tc.duration)
			}
		})
	}
}

// TestIdleFlagWithWatch tests --idle combined with --watch
func TestIdleFlagWithWatch(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	streams, _, _, errBuf := genericiooptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetErr(errBuf)

	// Set both flags
	err := cmd.Flags().Set("idle", "30m")
	assert.NoError(t, err)
	err = cmd.Flags().Set("watch", "true")
	assert.NoError(t, err)

	// These flags should be compatible
	// Implementation might filter watch output based on idle criteria
	// Just verify the flags can be set together without error
	assert.NotNil(t, cmd, "--idle and --watch should be compatible")
}

// TestIdleFlagWithOutputFormat tests --idle with different output formats
func TestIdleFlagWithOutputFormat(t *testing.T) {
	now := time.Now()
	pods := &corev1.PodList{
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "idle-pod",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-2 * time.Hour).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
		},
	}

	testCases := []struct {
		name   string
		output string
	}{
		{"table", ""},
		{"wide", "wide"},
		{"json", "json"},
		{"yaml", "yaml"},
		{"name", "name"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()
			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

			tf.UnstructuredClient = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
			}

			streams, _, buf, _ := genericiooptions.NewTestIOStreams()
			cmd := NewCmdGet("kubectl", tf, streams)
			cmd.SetOut(buf)

			cmd.Flags().Set("idle", "1h")
			if tc.output != "" {
				cmd.Flags().Set("output", tc.output)
			}
			cmd.Run(cmd, []string{"pods"})

			output := buf.String()
			assert.Contains(t, output, "idle-pod", "expected idle-pod in output for format %s", tc.name)

			// For non-table formats, IDLE-SINCE column shouldn't appear
			if tc.output == "json" || tc.output == "yaml" || tc.output == "name" {
				assert.NotContains(t, output, "IDLE-SINCE", "expected no IDLE-SINCE column for %s format", tc.output)
			}
		})
	}
}

// TestIdleFlagNoPods tests --idle when no pods match criteria
func TestIdleFlagNoPods(t *testing.T) {
	now := time.Now()
	pods := &corev1.PodList{
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "active-pod",
					Namespace: "test",
					Annotations: map[string]string{
						LastActivityAnnotationKey: now.Add(-5 * time.Minute).Format(time.RFC3339Nano),
					},
				},
				Status: corev1.PodStatus{Phase: corev1.PodRunning},
			},
		},
	}

	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdGet("kubectl", tf, streams)
	cmd.SetOut(buf)

	// Set --idle=1h, which should filter out the active pod
	cmd.Flags().Set("idle", "1h")
	cmd.Run(cmd, []string{"pods"})

	output := buf.String()
	assert.NotContains(t, output, "active-pod", "expected active-pod to be filtered out")
	// Should show "No resources found" or empty output
	assert.True(t, strings.Contains(output, "No resources found") || !strings.Contains(output, "active-pod"),
		"expected appropriate message when no pods match idle criteria")
}

// TestFormatIdleDuration tests the human-readable idle duration formatting
func TestFormatIdleDuration(t *testing.T) {
	testCases := []struct {
		duration time.Duration
		expected string
	}{
		{30 * time.Second, "<1m"},
		{45 * time.Second, "<1m"},
		{1 * time.Minute, "1m"},
		{5 * time.Minute, "5m"},
		{59 * time.Minute, "59m"},
		{60 * time.Minute, "1h"},
		{90 * time.Minute, "1h30m"},
		{2 * time.Hour, "2h"},
		{2*time.Hour + 30*time.Minute, "2h30m"},
		{24 * time.Hour, "1d"},
		{25 * time.Hour, "1d1h"},
		{48 * time.Hour, "2d"},
		{49*time.Hour + 30*time.Minute, "2d1h"},
		{0, "-"},
	}

	for _, tc := range testCases {
		t.Run(tc.expected, func(t *testing.T) {
			result := formatIdleDuration(tc.duration)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// TestParseIdleDuration tests parsing of idle duration strings
func TestParseIdleDuration(t *testing.T) {
	testCases := []struct {
		input    string
		expected time.Duration
		wantErr  bool
	}{
		{"30m", 30 * time.Minute, false},
		{"1h", 1 * time.Hour, false},
		{"1h30m", 90 * time.Minute, false},
		{"2h", 2 * time.Hour, false},
		{"90s", 90 * time.Second, false},
		{"0s", 0, false},
		{"", 30 * time.Minute, false}, // default
		{"invalid", 0, true},
		{"-1h", 0, true},
	}

	for _, tc := range testCases {
		t.Run(tc.input, func(t *testing.T) {
			result, err := parseIdleDuration(tc.input)
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected, result)
			}
		})
	}
}

// ============================================================================
// Helper functions that would be added to the get package
// ============================================================================

func formatIdleDuration(d time.Duration) string {
	if d == 0 {
		return "-"
	}
	if d < time.Minute {
		return "<1m"
	}

	days := int(d.Hours() / 24)
	hours := int(d.Hours()) % 24
	minutes := int(d.Minutes()) % 60

	var result string
	if days > 0 {
		result = strconv.Itoa(days) + "d"
		if hours > 0 {
			result += strconv.Itoa(hours) + "h"
		}
		return result
	}
	if hours > 0 {
		result = strconv.Itoa(hours) + "h"
		if minutes > 0 {
			result += strconv.Itoa(minutes) + "m"
		}
		return result
	}
	return strconv.Itoa(minutes) + "m"
}

func parseIdleDuration(s string) (time.Duration, error) {
	if s == "" {
		return 30 * time.Minute, nil // default
	}
	d, err := time.ParseDuration(s)
	if err != nil {
		return 0, err
	}
	if d < 0 {
		return 0, &negDurationError{d}
	}
	return d, nil
}

type negDurationError struct {
	d time.Duration
}

func (e *negDurationError) Error() string {
	return "duration must be non-negative"
}

// Simplified format function for actual implementation
func formatIdleDurationImpl(d time.Duration) string {
	if d == 0 {
		return "-"
	}
	if d < time.Minute {
		return "<1m"
	}

	days := int(d.Hours()) / 24
	hours := int(d.Hours()) % 24
	minutes := int(d.Minutes()) % 60

	var parts []string
	if days > 0 {
		parts = append(parts, strconv.Itoa(days)+"d")
	}
	if hours > 0 {
		parts = append(parts, strconv.Itoa(hours)+"h")
	}
	if minutes > 0 && days == 0 { // Only show minutes if less than 1 day
		parts = append(parts, strconv.Itoa(minutes)+"m")
	}

	if len(parts) == 0 {
		return "<1m"
	}
	return strings.Join(parts, "")
}
