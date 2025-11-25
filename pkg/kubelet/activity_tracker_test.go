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

package kubelet

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

// TestActivityTracker_RecordActivity tests that activity is properly recorded for pods
func TestActivityTracker_RecordActivity(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tracker := NewActivityTracker(fakeClock)

	podUID := types.UID("test-pod-123")

	// Initially, no activity should be recorded
	_, exists := tracker.GetLastActivity(podUID)
	assert.False(t, exists, "expected no activity initially")

	// Record activity
	tracker.RecordActivity(podUID, ActivityTypeExec)

	// Verify activity was recorded
	lastActivity, exists := tracker.GetLastActivity(podUID)
	assert.True(t, exists, "expected activity to be recorded")
	assert.Equal(t, fakeClock.Now(), lastActivity.Time)
}

// TestActivityTracker_RecordActivityUpdatesTimestamp verifies that recording activity updates the timestamp
func TestActivityTracker_RecordActivityUpdatesTimestamp(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tracker := NewActivityTracker(fakeClock)

	podUID := types.UID("test-pod-123")

	// Record initial activity
	tracker.RecordActivity(podUID, ActivityTypeExec)
	firstActivity, _ := tracker.GetLastActivity(podUID)

	// Advance time
	fakeClock.Step(5 * time.Minute)

	// Record new activity
	tracker.RecordActivity(podUID, ActivityTypePortForward)
	secondActivity, _ := tracker.GetLastActivity(podUID)

	assert.True(t, secondActivity.Time.After(firstActivity.Time),
		"expected second activity timestamp to be after first")
	assert.Equal(t, 5*time.Minute, secondActivity.Time.Sub(firstActivity.Time))
}

// TestActivityTracker_DifferentActivityTypes verifies different activity types are tracked
func TestActivityTracker_DifferentActivityTypes(t *testing.T) {
	testCases := []struct {
		name         string
		activityType ActivityType
	}{
		{"Exec", ActivityTypeExec},
		{"PortForward", ActivityTypePortForward},
		{"Logs", ActivityTypeLogs},
		{"Attach", ActivityTypeAttach},
		{"Copy", ActivityTypeCopy},
		{"HTTPRequest", ActivityTypeHTTPRequest},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			fakeClock := testingclock.NewFakeClock(time.Now())
			tracker := NewActivityTracker(fakeClock)
			podUID := types.UID("test-pod-" + tc.name)

			tracker.RecordActivity(podUID, tc.activityType)

			lastActivity, exists := tracker.GetLastActivity(podUID)
			assert.True(t, exists, "expected activity to be recorded for %s", tc.name)
			assert.Equal(t, fakeClock.Now(), lastActivity.Time)
		})
	}
}

// TestActivityTracker_GetIdlePods returns pods that have been idle for a given duration
func TestActivityTracker_GetIdlePods(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tracker := NewActivityTracker(fakeClock)

	pod1UID := types.UID("pod-1")
	pod2UID := types.UID("pod-2")
	pod3UID := types.UID("pod-3")

	// Record activity for all pods at different times
	tracker.RecordActivity(pod1UID, ActivityTypeExec)
	fakeClock.Step(10 * time.Minute)
	tracker.RecordActivity(pod2UID, ActivityTypeExec)
	fakeClock.Step(10 * time.Minute)
	tracker.RecordActivity(pod3UID, ActivityTypeExec)

	// Now: pod1 is 20 min idle, pod2 is 10 min idle, pod3 is 0 min idle
	idlePods := tracker.GetIdlePods(15 * time.Minute)

	assert.Len(t, idlePods, 1, "expected 1 idle pod")
	assert.Contains(t, idlePods, pod1UID, "expected pod1 to be idle")

	// Check for pods idle for 5 minutes
	idlePods = tracker.GetIdlePods(5 * time.Minute)
	assert.Len(t, idlePods, 2, "expected 2 idle pods")
	assert.Contains(t, idlePods, pod1UID)
	assert.Contains(t, idlePods, pod2UID)
}

// TestActivityTracker_GetIdlePodsMap returns a map of idle pods with their last activity time
func TestActivityTracker_GetIdlePodsMap(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tracker := NewActivityTracker(fakeClock)

	pod1UID := types.UID("pod-1")
	pod2UID := types.UID("pod-2")

	startTime := fakeClock.Now()

	tracker.RecordActivity(pod1UID, ActivityTypeExec)
	fakeClock.Step(30 * time.Minute)
	tracker.RecordActivity(pod2UID, ActivityTypeExec)

	idleMap := tracker.GetIdlePodsMap()

	require.Len(t, idleMap, 2)
	assert.Equal(t, startTime, idleMap[pod1UID].Time)
	assert.Equal(t, startTime.Add(30*time.Minute), idleMap[pod2UID].Time)
}

// TestActivityTracker_RemovePod tests that pods can be removed from tracking
func TestActivityTracker_RemovePod(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tracker := NewActivityTracker(fakeClock)

	podUID := types.UID("test-pod-123")

	tracker.RecordActivity(podUID, ActivityTypeExec)
	_, exists := tracker.GetLastActivity(podUID)
	assert.True(t, exists, "expected activity to exist before removal")

	tracker.RemovePod(podUID)

	_, exists = tracker.GetLastActivity(podUID)
	assert.False(t, exists, "expected activity to be removed")
}

// TestActivityTracker_ConcurrentAccess tests concurrent access to the tracker
func TestActivityTracker_ConcurrentAccess(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tracker := NewActivityTracker(fakeClock)

	done := make(chan bool, 100)

	// Start multiple goroutines recording activity
	for i := 0; i < 50; i++ {
		go func(idx int) {
			podUID := types.UID("pod-" + string(rune(idx)))
			tracker.RecordActivity(podUID, ActivityTypeExec)
			done <- true
		}(i)
	}

	// Start multiple goroutines reading activity
	for i := 0; i < 50; i++ {
		go func(idx int) {
			podUID := types.UID("pod-" + string(rune(idx)))
			tracker.GetLastActivity(podUID)
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 100; i++ {
		<-done
	}
}

// TestActivityTracker_PersistAndRestore tests persistence to annotations
func TestActivityTracker_PersistAndRestore(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tracker := NewActivityTracker(fakeClock)

	podUID := types.UID("test-pod-123")
	tracker.RecordActivity(podUID, ActivityTypeExec)

	// Get annotation value
	annotationValue := tracker.GetAnnotationValue(podUID)
	assert.NotEmpty(t, annotationValue, "expected annotation value to be non-empty")

	// Create new tracker and restore from annotation
	newTracker := NewActivityTracker(fakeClock)
	err := newTracker.RestoreFromAnnotation(podUID, annotationValue)
	require.NoError(t, err)

	// Verify restored activity matches
	original, _ := tracker.GetLastActivity(podUID)
	restored, exists := newTracker.GetLastActivity(podUID)
	assert.True(t, exists, "expected restored activity to exist")
	assert.Equal(t, original.Time, restored.Time, "expected timestamps to match")
}

// TestActivityTracker_GetIdleDuration calculates idle duration correctly
func TestActivityTracker_GetIdleDuration(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tracker := NewActivityTracker(fakeClock)

	podUID := types.UID("test-pod-123")
	tracker.RecordActivity(podUID, ActivityTypeExec)

	fakeClock.Step(25 * time.Minute)

	duration, exists := tracker.GetIdleDuration(podUID)
	assert.True(t, exists)
	assert.Equal(t, 25*time.Minute, duration)
}

// TestActivityTracker_NeverActiveReturnsZero tests pods that were never active
func TestActivityTracker_NeverActiveReturnsZero(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tracker := NewActivityTracker(fakeClock)

	podUID := types.UID("never-active-pod")

	_, exists := tracker.GetIdleDuration(podUID)
	assert.False(t, exists, "expected no duration for never-active pod")
}

// TestActivityTracker_FormatIdleDuration tests human-readable idle duration formatting
func TestActivityTracker_FormatIdleDuration(t *testing.T) {
	testCases := []struct {
		duration time.Duration
		expected string
	}{
		{30 * time.Second, "<1m"},
		{1 * time.Minute, "1m"},
		{5 * time.Minute, "5m"},
		{59 * time.Minute, "59m"},
		{1 * time.Hour, "1h"},
		{2 * time.Hour, "2h"},
		{90 * time.Minute, "1h30m"},
		{25 * time.Hour, "1d1h"},
		{48 * time.Hour, "2d"},
		{0, "-"},
	}

	for _, tc := range testCases {
		t.Run(tc.expected, func(t *testing.T) {
			result := FormatIdleDuration(tc.duration)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// ActivityType represents the type of activity that triggered the timestamp update
type ActivityType string

const (
	ActivityTypeExec        ActivityType = "exec"
	ActivityTypePortForward ActivityType = "port-forward"
	ActivityTypeLogs        ActivityType = "logs"
	ActivityTypeAttach      ActivityType = "attach"
	ActivityTypeCopy        ActivityType = "copy"
	ActivityTypeHTTPRequest ActivityType = "http-request"
)

// ActivityTracker tracks the last activity time for each pod
type ActivityTracker struct {
	clock      clock.Clock
	activities map[types.UID]metav1.Time
}

// NewActivityTracker creates a new ActivityTracker
func NewActivityTracker(clk clock.Clock) *ActivityTracker {
	return &ActivityTracker{
		clock:      clk,
		activities: make(map[types.UID]metav1.Time),
	}
}

// RecordActivity records an activity for a pod
func (t *ActivityTracker) RecordActivity(podUID types.UID, activityType ActivityType) {
	t.activities[podUID] = metav1.NewTime(t.clock.Now())
}

// GetLastActivity returns the last activity time for a pod
func (t *ActivityTracker) GetLastActivity(podUID types.UID) (metav1.Time, bool) {
	activity, exists := t.activities[podUID]
	return activity, exists
}

// GetIdlePods returns UIDs of pods that have been idle for at least the given duration
func (t *ActivityTracker) GetIdlePods(idleDuration time.Duration) []types.UID {
	var idlePods []types.UID
	cutoff := t.clock.Now().Add(-idleDuration)
	for uid, activity := range t.activities {
		if activity.Time.Before(cutoff) {
			idlePods = append(idlePods, uid)
		}
	}
	return idlePods
}

// GetIdlePodsMap returns a map of all pod UIDs to their last activity time
func (t *ActivityTracker) GetIdlePodsMap() map[types.UID]metav1.Time {
	result := make(map[types.UID]metav1.Time)
	for uid, activity := range t.activities {
		result[uid] = activity
	}
	return result
}

// RemovePod removes a pod from tracking
func (t *ActivityTracker) RemovePod(podUID types.UID) {
	delete(t.activities, podUID)
}

// GetAnnotationValue returns the annotation value for persisting the last activity time
func (t *ActivityTracker) GetAnnotationValue(podUID types.UID) string {
	activity, exists := t.activities[podUID]
	if !exists {
		return ""
	}
	return activity.Format(time.RFC3339Nano)
}

// RestoreFromAnnotation restores the last activity time from an annotation value
func (t *ActivityTracker) RestoreFromAnnotation(podUID types.UID, value string) error {
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return err
	}
	t.activities[podUID] = metav1.NewTime(parsed)
	return nil
}

// GetIdleDuration returns how long a pod has been idle
func (t *ActivityTracker) GetIdleDuration(podUID types.UID) (time.Duration, bool) {
	activity, exists := t.activities[podUID]
	if !exists {
		return 0, false
	}
	return t.clock.Now().Sub(activity.Time), true
}

// FormatIdleDuration formats a duration into a human-readable idle time string
func FormatIdleDuration(d time.Duration) string {
	if d == 0 {
		return "-"
	}
	if d < time.Minute {
		return "<1m"
	}

	days := int(d.Hours() / 24)
	hours := int(d.Hours()) % 24
	minutes := int(d.Minutes()) % 60

	if days > 0 {
		if hours > 0 {
			return time.Duration(days*24 + hours).Truncate(time.Hour).String()[:len(time.Duration(days*24+hours).Truncate(time.Hour).String())-2] + "d" + time.Duration(hours).Truncate(time.Hour).String()[:len(time.Duration(hours).Truncate(time.Hour).String())-2] + "h"
		}
		return time.Duration(days * 24).Truncate(time.Hour).String()[:len(time.Duration(days*24).Truncate(time.Hour).String())-2] + "d"
	}
	if hours > 0 {
		if minutes > 0 {
			return time.Duration(hours).Truncate(time.Hour).String()[:len(time.Duration(hours).Truncate(time.Hour).String())-2] + "h" + time.Duration(minutes).Truncate(time.Minute).String()[:len(time.Duration(minutes).Truncate(time.Minute).String())-2] + "m"
		}
		return time.Duration(hours).Truncate(time.Hour).String()[:len(time.Duration(hours).Truncate(time.Hour).String())-2] + "h"
	}
	return time.Duration(minutes).Truncate(time.Minute).String()[:len(time.Duration(minutes).Truncate(time.Minute).String())-2] + "m"
}
