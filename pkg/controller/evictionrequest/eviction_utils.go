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

package evictionrequest

import (
	"slices"
	"strings"
	"time"

	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
)

// hasCompleted returns true if the Eviction has reached
// a terminal state (Canceled or Evicted condition is True).
func hasEvictionCompleted(eviction *lifecyclev1alpha1.Eviction) bool {
	return hasEvictionSucceeded(eviction) || hasEvictionFailed(eviction)
}

func hasEvictionSucceeded(eviction *lifecyclev1alpha1.Eviction) bool {
	return meta.IsStatusConditionTrue(eviction.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionTargetEvicted))
}

func hasEvictionFailed(eviction *lifecyclev1alpha1.Eviction) bool {
	return meta.IsStatusConditionTrue(eviction.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionFailed))
}

// getOrInitializeTargetResponders initializers the target responders list.
// Returns the existing list if already initialized, or a new list from the target's eviction responders.
// The second value returns true if the list is initialized for the first time.
func getOrInitializeTargetResponders(eviction *lifecyclev1alpha1.Eviction, target targetInfo) ([]lifecyclev1alpha1.TargetResponder, bool) {
	// TargetResponders entries cannot be added or removed after first initialization
	if len(eviction.Status.TargetResponders) > 0 {
		targets := make([]lifecyclev1alpha1.TargetResponder, len(eviction.Status.TargetResponders))
		copy(targets, eviction.Status.TargetResponders)
		return targets, false
	}

	responders := target.evictionResponders(true)
	if len(responders) == 0 {
		return nil, false
	}
	targets := make([]lifecyclev1alpha1.TargetResponder, 0, len(responders))
	for _, responder := range responders {
		targets = append(targets, lifecyclev1alpha1.TargetResponder{
			Name:     responder.Name,
			Priority: responder.Priority,
			State:    lifecyclev1alpha1.ResponderStateInactive,
		})
	}
	return targets, true
}

// computeResponderProgression computes the progression of responder states.
func computeResponderProgression(now time.Time, eviction *lifecyclev1alpha1.Eviction, targetResponders []lifecyclev1alpha1.TargetResponder, target targetInfo, targetIsGone, targetIsTerminal, evictionIsCanceled bool) (bool, *time.Duration) {
	// No target responders: nothing to process
	if len(targetResponders) == 0 {
		return true, nil
	}
	switch targetResponders[len(targetResponders)-1].State {
	case lifecyclev1alpha1.ResponderStateInterrupted,
		lifecyclev1alpha1.ResponderStateCanceled,
		lifecyclev1alpha1.ResponderStateCompleted:
		// no other progression possible
		return true, nil
	}
	activeIdx := findTargetResponderIdx(targetResponders, lifecyclev1alpha1.ResponderStateActive)
	activeResponderNotFound := activeIdx == -1
	switch {
	case targetIsGone || targetIsTerminal:
		if activeResponderNotFound {
			// all responder work is done - do not start a new one
			return false, nil
		}
		activeResponderStatus := findResponderStatus(eviction.Status.Responders, targetResponders[activeIdx].Name)
		if activeResponderStatus != nil && activeResponderStatus.CompletionTime != nil {
			// successful completion
			targetResponders[activeIdx].State = lifecyclev1alpha1.ResponderStateCompleted
			return false, nil
		}
		if deferForResponderUpdate := shouldDeferCompletion(now, activeResponderStatus, target); deferForResponderUpdate != nil {
			// the responder might report status later
			return false, deferForResponderUpdate
		}
		// responder got stuck reporting the completion time
		targetResponders[activeIdx].State = lifecyclev1alpha1.ResponderStateInterrupted
		return false, nil
	case evictionIsCanceled:
		if activeResponderNotFound {
			// all responder work is done - do not start a new one
			return false, nil
		}
		// canceled
		targetResponders[activeIdx].State = lifecyclev1alpha1.ResponderStateCanceled
		return false, nil
	}

	// activate the first responder
	if activeResponderNotFound {
		activeIdx = findTargetResponderIdx(targetResponders, lifecyclev1alpha1.ResponderStateInactive)
	}

	activeResponderStatus := findResponderStatus(eviction.Status.Responders, targetResponders[activeIdx].Name)
	assignedResponderState, resyncAfter := computeResponderStateAndNextResync(now, activeResponderStatus)
	targetResponders[activeIdx].State = assignedResponderState
	if assignedResponderState != lifecyclev1alpha1.ResponderStateActive && activeIdx+1 < len(targetResponders) {
		// activate the next one
		targetResponders[activeIdx+1].State = lifecyclev1alpha1.ResponderStateActive
		resyncAfter = new(ResponderHeartbeatTimeout)
	}
	return false, resyncAfter
}

// computeResponderStateAndNextResync determines if we should advance from the current responder.
// Returns (current responder state, resyncAfter). If not advancing, resyncAfter indicates when to check again.
func computeResponderStateAndNextResync(now time.Time, status *lifecyclev1alpha1.ResponderStatus) (lifecyclev1alpha1.ResponderStateType, *time.Duration) {
	// First sync, advance as there is no current active responder
	if status == nil {
		return lifecyclev1alpha1.ResponderStateActive, new(ResponderHeartbeatTimeout)
	}

	// Advance as responder has completed
	if status.CompletionTime != nil {
		return lifecyclev1alpha1.ResponderStateCompleted, nil
	}
	// If there is no startTime, we will set it during the same sync, so we can set now here.
	lastUpdate := now
	if status.StartTime != nil {
		lastUpdate = status.StartTime.Time
	}
	if status.HeartbeatTime != nil {
		lastUpdate = status.HeartbeatTime.Time
	}

	elapsed := now.Sub(lastUpdate)
	// Advance as heartbeat timeout has been reached
	if elapsed >= ResponderHeartbeatTimeout {
		return lifecyclev1alpha1.ResponderStateInterrupted, nil
	}
	// Schedule resync when timeout would occur
	return lifecyclev1alpha1.ResponderStateActive, new(ResponderHeartbeatTimeout - elapsed)

}

func sortEvictionRequestsByRelevance(sortedRequests []*lifecyclev1alpha1.EvictionRequest) {
	slices.SortStableFunc(sortedRequests, func(a *lifecyclev1alpha1.EvictionRequest, b *lifecyclev1alpha1.EvictionRequest) int {
		aExists := !a.CreationTimestamp.IsZero()
		bExists := !b.CreationTimestamp.IsZero()
		aDeleted := !aExists || a.DeletionTimestamp != nil
		bDeleted := !bExists || b.DeletionTimestamp != nil
		// Prefer existing EvictionRequest objects over old eviction.Status.Requesters
		if aExists && !bExists {
			return -1
		}
		if !aExists && bExists {
			return 1
		}
		// Prefer non deleted EvictionRequest (deleted are considered withdrawn)
		if !aDeleted && bDeleted {
			return -1
		}
		if aDeleted && !bDeleted {
			return 1
		}
		// Prefer eviction intents over withdrawn.
		if a.Spec.Intent != b.Spec.Intent && a.Spec.Intent == lifecyclev1alpha1.EvictionRequestIntentEviction && !aDeleted { // Deleted default to Withdrawn.
			return -1
		}
		if a.Spec.Intent != b.Spec.Intent && b.Spec.Intent == lifecyclev1alpha1.EvictionRequestIntentEviction && !bDeleted { // Deleted default to Withdrawn.
			return 1
		}
		// Prefer oldest since they are already present in the status, so we don't do unnecessary updates
		cmp := a.CreationTimestamp.Time.Compare(b.CreationTimestamp.Time)
		if cmp != 0 {
			return cmp
		}
		// Compare names if the timestamp is the same.
		return strings.Compare(a.Spec.Requester, b.Spec.Requester)
	})
}
