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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// LastActivityAnnotationKey is the annotation key for storing last activity time
const LastActivityAnnotationKey = "kubernetes.io/last-activity"

// TestLastActivityTracker is a simplified activity tracker for testing
type TestLastActivityTracker struct {
	mu         sync.RWMutex
	activities map[types.UID]metav1.Time
}

// NewTestLastActivityTracker creates a new tracker
func NewTestLastActivityTracker() *TestLastActivityTracker {
	return &TestLastActivityTracker{
		activities: make(map[types.UID]metav1.Time),
	}
}

// SetLastActivityTime sets the last activity time for a pod
func (t *TestLastActivityTracker) SetLastActivityTime(podUID types.UID, activityTime metav1.Time) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.activities[podUID] = activityTime
}

// GetLastActivityTime returns the last activity time for a pod
func (t *TestLastActivityTracker) GetLastActivityTime(podUID types.UID) (metav1.Time, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	activity, exists := t.activities[podUID]
	return activity, exists
}

// RecordPodActivity records an activity event for a pod
func (t *TestLastActivityTracker) RecordPodActivity(podUID types.UID, activityType string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.activities[podUID] = metav1.Now()
}

// GetIdlePods returns UIDs of pods idle for the given duration
func (t *TestLastActivityTracker) GetIdlePods(idleDuration time.Duration) []types.UID {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var idlePods []types.UID
	cutoff := time.Now().Add(-idleDuration)

	for uid, activity := range t.activities {
		if activity.Time.Before(cutoff) {
			idlePods = append(idlePods, uid)
		}
	}
	return idlePods
}

// GetIdlePodsMap returns a map of pod UIDs to their last activity times
func (t *TestLastActivityTracker) GetIdlePodsMap() map[types.UID]metav1.Time {
	t.mu.RLock()
	defer t.mu.RUnlock()

	result := make(map[types.UID]metav1.Time)
	for uid, activity := range t.activities {
		result[uid] = activity
	}
	return result
}

// ClearLastActivityTime removes the last activity time for a pod
func (t *TestLastActivityTracker) ClearLastActivityTime(podUID types.UID) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.activities, podUID)
}

// RestoreFromAnnotation restores activity time from pod annotation
func (t *TestLastActivityTracker) RestoreFromAnnotation(pod *v1.Pod) {
	annotation, exists := pod.Annotations[LastActivityAnnotationKey]
	if !exists {
		return
	}

	parsed, err := time.Parse(time.RFC3339Nano, annotation)
	if err != nil {
		return
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	t.activities[pod.UID] = metav1.NewTime(parsed)
}

// GetAnnotationValue returns the annotation value for persisting the last activity time
func (t *TestLastActivityTracker) GetAnnotationValue(podUID types.UID) string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	activity, exists := t.activities[podUID]
	if !exists {
		return ""
	}
	return activity.Format(time.RFC3339Nano)
}

// ============================================================================
// Tests
// ============================================================================

// TestSetLastActivityTime tests setting the last activity time on a pod
func TestSetLastActivityTime(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("test-pod-123")
	now := metav1.Now()

	// Set the last activity time
	tracker.SetLastActivityTime(podUID, now)

	// Verify it was stored
	stored, exists := tracker.GetLastActivityTime(podUID)
	require.True(t, exists, "expected activity time to exist")
	assert.Equal(t, now.Time.Unix(), stored.Time.Unix())
}

// TestUpdateLastActivityTimeOnExec verifies activity time updates on exec
func TestUpdateLastActivityTimeOnExec(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("test-pod-123")

	// Initially no activity
	_, exists := tracker.GetLastActivityTime(podUID)
	assert.False(t, exists, "expected no activity initially")

	// Simulate exec activity
	tracker.RecordPodActivity(podUID, "exec")

	// Verify activity was recorded
	lastActivity, exists := tracker.GetLastActivityTime(podUID)
	assert.True(t, exists, "expected activity time to be recorded")
	assert.False(t, lastActivity.IsZero(), "expected non-zero activity time")
}

// TestUpdateLastActivityTimeOnPortForward verifies activity time updates on port-forward
func TestUpdateLastActivityTimeOnPortForward(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("test-pod-123")

	tracker.RecordPodActivity(podUID, "port-forward")

	lastActivity, exists := tracker.GetLastActivityTime(podUID)
	assert.True(t, exists, "expected activity time to be recorded")
	assert.False(t, lastActivity.IsZero(), "expected non-zero activity time")
}

// TestUpdateLastActivityTimeOnLogs verifies activity time updates on logs
func TestUpdateLastActivityTimeOnLogs(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("test-pod-123")

	tracker.RecordPodActivity(podUID, "logs")

	lastActivity, exists := tracker.GetLastActivityTime(podUID)
	assert.True(t, exists, "expected activity time to be recorded")
	assert.False(t, lastActivity.IsZero(), "expected non-zero activity time")
}

// TestUpdateLastActivityTimeOnAttach verifies activity time updates on attach
func TestUpdateLastActivityTimeOnAttach(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("test-pod-123")

	tracker.RecordPodActivity(podUID, "attach")

	lastActivity, exists := tracker.GetLastActivityTime(podUID)
	assert.True(t, exists, "expected activity time to be recorded")
	assert.False(t, lastActivity.IsZero(), "expected non-zero activity time")
}

// TestLastActivityTimeSurvivesRestart tests persistence via annotations
func TestLastActivityTimeSurvivesRestart(t *testing.T) {
	tracker1 := NewTestLastActivityTracker()

	podUID := types.UID("test-pod-123")
	activityTime := metav1.NewTime(time.Now().Add(-30 * time.Minute))

	tracker1.SetLastActivityTime(podUID, activityTime)

	// Get annotation value
	annotationValue := tracker1.GetAnnotationValue(podUID)
	assert.NotEmpty(t, annotationValue, "expected annotation value to be non-empty")

	// Simulate restart by creating new tracker and restoring from annotation
	podWithAnnotation := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:  podUID,
			Name: "test-pod",
			Annotations: map[string]string{
				LastActivityAnnotationKey: annotationValue,
			},
		},
	}

	tracker2 := NewTestLastActivityTracker()
	tracker2.RestoreFromAnnotation(podWithAnnotation)

	// Verify the activity time was restored
	restoredTime, exists := tracker2.GetLastActivityTime(podUID)
	require.True(t, exists, "expected activity time to be restored")
	assert.Equal(t, activityTime.Time.Unix(), restoredTime.Time.Unix(), "expected restored time to match original")
}

// TestGetIdlePods returns pods idle for a given duration
func TestGetIdlePods(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	now := time.Now()
	pod1UID := types.UID("pod-1")
	pod2UID := types.UID("pod-2")
	pod3UID := types.UID("pod-3")

	tracker.SetLastActivityTime(pod1UID, metav1.NewTime(now.Add(-2*time.Hour)))    // 2 hours idle
	tracker.SetLastActivityTime(pod2UID, metav1.NewTime(now.Add(-30*time.Minute))) // 30 min idle
	tracker.SetLastActivityTime(pod3UID, metav1.NewTime(now.Add(-5*time.Minute)))  // 5 min idle

	// Get pods idle for 1 hour
	idlePods := tracker.GetIdlePods(1 * time.Hour)
	assert.Len(t, idlePods, 1, "expected 1 pod idle for 1 hour")
	assert.Contains(t, idlePods, pod1UID)

	// Get pods idle for 15 minutes
	idlePods = tracker.GetIdlePods(15 * time.Minute)
	assert.Len(t, idlePods, 2, "expected 2 pods idle for 15 minutes")
	assert.Contains(t, idlePods, pod1UID)
	assert.Contains(t, idlePods, pod2UID)
}

// TestGetIdlePodsWithActivityMap returns a map of pod UIDs to last activity times
func TestGetIdlePodsWithActivityMap(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	pod1UID := types.UID("pod-1")
	now := time.Now()
	activityTime := metav1.NewTime(now.Add(-45 * time.Minute))
	tracker.SetLastActivityTime(pod1UID, activityTime)

	idleMap := tracker.GetIdlePodsMap()
	require.Len(t, idleMap, 1)
	assert.Equal(t, activityTime.Time.Unix(), idleMap[pod1UID].Time.Unix())
}

// TestActivityTimeNotSetForNewPods verifies new pods have no activity time
func TestActivityTimeNotSetForNewPods(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("new-pod")

	_, exists := tracker.GetLastActivityTime(podUID)
	assert.False(t, exists, "new pod should not have activity time")
}

// TestClearLastActivityTime verifies activity time can be cleared
func TestClearLastActivityTime(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("test-pod-123")
	now := metav1.Now()

	tracker.SetLastActivityTime(podUID, now)
	_, exists := tracker.GetLastActivityTime(podUID)
	assert.True(t, exists, "expected activity time to exist")

	tracker.ClearLastActivityTime(podUID)
	_, exists = tracker.GetLastActivityTime(podUID)
	assert.False(t, exists, "expected activity time to be cleared")
}

// TestConcurrentActivityUpdates tests concurrent access to the tracker
func TestConcurrentActivityUpdates(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	done := make(chan bool, 100)

	// Start multiple goroutines recording activity
	for i := 0; i < 50; i++ {
		go func(idx int) {
			podUID := types.UID("pod-" + string(rune('a'+idx%26)))
			tracker.RecordPodActivity(podUID, "exec")
			done <- true
		}(i)
	}

	// Start multiple goroutines reading activity
	for i := 0; i < 50; i++ {
		go func(idx int) {
			podUID := types.UID("pod-" + string(rune('a'+idx%26)))
			tracker.GetLastActivityTime(podUID)
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 100; i++ {
		<-done
	}

	// Should not panic or have race conditions
}

// TestActivityTimeUpdateOverwrite verifies newer activity overwrites older
func TestActivityTimeUpdateOverwrite(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("test-pod-123")

	// Set initial activity
	oldTime := metav1.NewTime(time.Now().Add(-1 * time.Hour))
	tracker.SetLastActivityTime(podUID, oldTime)

	// Get the old activity
	storedOld, _ := tracker.GetLastActivityTime(podUID)

	// Set new activity
	time.Sleep(10 * time.Millisecond) // Ensure time advances
	tracker.RecordPodActivity(podUID, "exec")

	// Verify new activity is more recent
	storedNew, _ := tracker.GetLastActivityTime(podUID)
	assert.True(t, storedNew.Time.After(storedOld.Time),
		"new activity should be more recent than old activity")
}

// TestRestoreFromAnnotationInvalidFormat handles invalid annotation format
func TestRestoreFromAnnotationInvalidFormat(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("test-pod")
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:  podUID,
			Name: "test-pod",
			Annotations: map[string]string{
				LastActivityAnnotationKey: "invalid-timestamp-format",
			},
		},
	}

	// Should not panic on invalid format
	tracker.RestoreFromAnnotation(pod)

	// Should not have any activity set (invalid format ignored)
	_, exists := tracker.GetLastActivityTime(podUID)
	assert.False(t, exists, "invalid timestamp should not be stored")
}

// TestRestoreFromAnnotationMissingAnnotation handles missing annotation
func TestRestoreFromAnnotationMissingAnnotation(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	podUID := types.UID("test-pod")
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:  podUID,
			Name: "test-pod",
			// No annotations
		},
	}

	// Should not panic on missing annotation
	tracker.RestoreFromAnnotation(pod)

	// Should not have any activity set
	_, exists := tracker.GetLastActivityTime(podUID)
	assert.False(t, exists, "no annotation should result in no activity")
}

// TestMultiplePodsTracking tests tracking activity for multiple pods
func TestMultiplePodsTracking(t *testing.T) {
	tracker := NewTestLastActivityTracker()

	numPods := 100
	for i := 0; i < numPods; i++ {
		podUID := types.UID("pod-" + string(rune('0'+i/10)) + string(rune('0'+i%10)))
		tracker.RecordPodActivity(podUID, "exec")
	}

	// Verify all pods have activity
	allActivities := tracker.GetIdlePodsMap()
	assert.Len(t, allActivities, numPods, "expected all pods to have activity")
}
