/*
Copyright The Kubernetes Authors.

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

package imperativeevictionresponder

import (
	"math"
	"regexp"
	"strconv"
	"sync"
	"time"

	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	apimachinerytypes "k8s.io/apimachinery/pkg/types"
	lifecycleapplyv1alpha1 "k8s.io/client-go/applyconfigurations/lifecycle/v1alpha1"
	"k8s.io/utils/ptr"
)

var attemptsRegex = regexp.MustCompile(`\(attempts=(\d+)\)`)

func exponentialBackoff(baseDelay, maxDelay time.Duration, exp uint64) time.Duration {
	// The backoff is capped such that 'calculated' value never overflows.
	backoff := float64(baseDelay.Nanoseconds()) * math.Pow(2, float64(exp))
	if backoff > math.MaxInt64 {
		return maxDelay
	}

	calculated := time.Duration(backoff)
	if calculated > maxDelay {
		return maxDelay
	}

	return calculated
}

func shouldHandleEviction(eviction *lifecyclev1alpha1.Eviction) bool {
	if eviction.Spec.Target.Pod == nil {
		return false
	}
	responderRole, ok := eviction.Labels[lifecyclev1alpha1.EvictionResponderImperativeEviction]
	if !ok {
		return false
	}
	switch responderRole {
	case string(lifecyclev1alpha1.EvictionParticipantRoleResponder), string(lifecyclev1alpha1.EvictionParticipantRoleRequesterResponder):
		return true
	}
	if meta.IsStatusConditionTrue(eviction.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionFailed)) ||
		meta.IsStatusConditionTrue(eviction.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionTargetEvicted)) {
		// Eviction final state reached: failed/canceled or done.
		return false
	}
	return false
}

func hasLabelChanged(oldEviction, eviction *lifecyclev1alpha1.Eviction, labelKey string) bool {
	oldLabelValue, oldOk := oldEviction.Labels[labelKey]
	labelValue, ok := eviction.Labels[labelKey]
	return oldOk != ok || oldLabelValue != labelValue
}

func findTargetResponderStatus(eviction *lifecyclev1alpha1.Eviction) *lifecyclev1alpha1.TargetResponder {
	for i, responder := range eviction.Status.TargetResponders {
		if responder.Name == lifecyclev1alpha1.EvictionResponderImperativeEviction {
			return &eviction.Status.TargetResponders[i]
		}
	}
	return nil
}

func findResponderStatus(eviction *lifecyclev1alpha1.Eviction) *lifecyclev1alpha1.ResponderStatus {
	for i, responder := range eviction.Status.Responders {
		if responder.Name == lifecyclev1alpha1.EvictionResponderImperativeEviction {
			return &eviction.Status.Responders[i]
		}
	}
	return nil
}

func getRecordedAttempts(message string) uint64 {
	subMatches := attemptsRegex.FindStringSubmatch(message)
	if len(subMatches) != 2 {
		return 0
	}
	if attempts, err := strconv.ParseUint(subMatches[1], 10, 64); err == nil {
		return attempts
	}
	return 0
}

func toResponderStatusApplyConfiguration(status lifecyclev1alpha1.ResponderStatus) *lifecycleapplyv1alpha1.ResponderStatusApplyConfiguration {
	result := lifecycleapplyv1alpha1.ResponderStatus().
		WithName(status.Name).
		WithMessage(ptr.Deref(status.Message, ""))
	if status.StartTime != nil {
		result.WithStartTime(*status.StartTime)
	}
	if status.HeartbeatTime != nil {
		result.WithHeartbeatTime(*status.HeartbeatTime)
	}
	if status.ExpectedCompletionTime != nil {
		result.WithExpectedCompletionTime(*status.ExpectedCompletionTime)
	}
	if status.CompletionTime != nil {
		result.WithCompletionTime(*status.CompletionTime)
	}
	return result
}

type lastEvictionAttempts struct {
	lock             sync.RWMutex
	evictionAttempts map[apimachinerytypes.UID]time.Time
}

func NewLastEvictionAttempts() *lastEvictionAttempts {
	return &lastEvictionAttempts{
		evictionAttempts: make(map[apimachinerytypes.UID]time.Time),
	}
}

func (l *lastEvictionAttempts) set(uid apimachinerytypes.UID, evictionAttempt time.Time) {
	l.lock.Lock()
	defer l.lock.Unlock()
	l.evictionAttempts[uid] = evictionAttempt
}

func (l *lastEvictionAttempts) remove(uid apimachinerytypes.UID) {
	l.lock.Lock()
	defer l.lock.Unlock()
	delete(l.evictionAttempts, uid)
}

// return the size of the list
func (l *lastEvictionAttempts) get(uid apimachinerytypes.UID) (time.Time, bool) {
	l.lock.RLock()
	defer l.lock.RUnlock()
	result, ok := l.evictionAttempts[uid]
	return result, ok
}
