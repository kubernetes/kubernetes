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
	"fmt"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/clock"
)

// LastActivityAnnotationKey is the annotation key used to persist the last activity time
const LastActivityAnnotationKey = "kubernetes.io/last-activity"

// ActivityType represents the type of activity that triggered the timestamp update
type ActivityType string

const (
	// ActivityTypeExec represents a kubectl exec activity
	ActivityTypeExec ActivityType = "exec"
	// ActivityTypePortForward represents a kubectl port-forward activity
	ActivityTypePortForward ActivityType = "port-forward"
	// ActivityTypeLogs represents a kubectl logs activity
	ActivityTypeLogs ActivityType = "logs"
	// ActivityTypeAttach represents a kubectl attach activity
	ActivityTypeAttach ActivityType = "attach"
	// ActivityTypeCopy represents a kubectl cp activity
	ActivityTypeCopy ActivityType = "copy"
	// ActivityTypeHTTPRequest represents an HTTP request to the container
	ActivityTypeHTTPRequest ActivityType = "http-request"
)

// ActivityTracker tracks the last activity time for each pod.
// It is safe for concurrent use.
type ActivityTracker struct {
	clock      clock.Clock
	mu         sync.RWMutex
	activities map[types.UID]metav1.Time
}

// NewActivityTracker creates a new ActivityTracker
func NewActivityTracker(clk clock.Clock) *ActivityTracker {
	return &ActivityTracker{
		clock:      clk,
		activities: make(map[types.UID]metav1.Time),
	}
}

// RecordActivity records an activity for a pod, updating its last activity timestamp
func (t *ActivityTracker) RecordActivity(podUID types.UID, activityType ActivityType) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.activities[podUID] = metav1.NewTime(t.clock.Now())
}

// GetLastActivity returns the last activity time for a pod
func (t *ActivityTracker) GetLastActivity(podUID types.UID) (metav1.Time, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	activity, exists := t.activities[podUID]
	return activity, exists
}

// GetIdlePods returns UIDs of pods that have been idle for at least the given duration
func (t *ActivityTracker) GetIdlePods(idleDuration time.Duration) []types.UID {
	t.mu.RLock()
	defer t.mu.RUnlock()
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
	t.mu.RLock()
	defer t.mu.RUnlock()
	result := make(map[types.UID]metav1.Time, len(t.activities))
	for uid, activity := range t.activities {
		result[uid] = activity
	}
	return result
}

// RemovePod removes a pod from tracking
func (t *ActivityTracker) RemovePod(podUID types.UID) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.activities, podUID)
}

// GetAnnotationValue returns the annotation value for persisting the last activity time
func (t *ActivityTracker) GetAnnotationValue(podUID types.UID) string {
	t.mu.RLock()
	defer t.mu.RUnlock()
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
		return fmt.Errorf("failed to parse activity timestamp: %w", err)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.activities[podUID] = metav1.NewTime(parsed)
	return nil
}

// GetIdleDuration returns how long a pod has been idle
func (t *ActivityTracker) GetIdleDuration(podUID types.UID) (time.Duration, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	activity, exists := t.activities[podUID]
	if !exists {
		return 0, false
	}
	return t.clock.Now().Sub(activity.Time), true
}

// FormatIdleDuration formats a duration into a human-readable idle time string.
// Returns "-" for zero duration, "<1m" for durations under a minute,
// and formats like "5m", "2h", "1h30m", "2d", "1d1h" for longer durations.
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
			return fmt.Sprintf("%dd%dh", days, hours)
		}
		return fmt.Sprintf("%dd", days)
	}
	if hours > 0 {
		if minutes > 0 {
			return fmt.Sprintf("%dh%dm", hours, minutes)
		}
		return fmt.Sprintf("%dh", hours)
	}
	return fmt.Sprintf("%dm", minutes)
}

