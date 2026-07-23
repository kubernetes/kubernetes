package evictionrequest

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	"k8s.io/kubernetes/pkg/apis/lifecycle"
	validation2 "k8s.io/kubernetes/pkg/apis/lifecycle/validation"
)

// newEvictionName returns the name in pod-1-foo format. If there are Evictions with a conflicting name,
// the counter is increased and pod-2-foo, etc. name is returned.
func newEvictionName(target targetInfo, allEvictions []*lifecyclev1alpha1.Eviction) string {
	nameCounter := 0

	if len(target.targetName()) == 0 {
		return ""
	}

	evictionNameRegex := regexp.MustCompile(fmt.Sprintf("%s-([0-9]+)-(.*)$", target.targetType().String()))
	for _, eviction := range allEvictions {
		if subMatches := evictionNameRegex.FindStringSubmatch(eviction.Name); len(subMatches) > 2 {
			counterPart := subMatches[1]
			namePart := subMatches[2]
			// name should fully match or partially if it reached the length limit
			// Thus it is possible for two different very long pod names to share a single counter space.
			matchesName := namePart == target.targetName() ||
				(len(eviction.Name) == validation.DNS1123SubdomainMaxLength && strings.HasPrefix(target.targetName(), namePart))
			if !matchesName {
				continue
			}
			if counter, err := strconv.Atoi(counterPart); err == nil && counter > nameCounter {
				nameCounter = counter
			}
		}
	}

	newName := fmt.Sprintf("%s-%d-%s", target.targetType().String(), nameCounter+1, target.targetName())
	if len(newName) > validation.DNS1123SubdomainMaxLength {
		newName = newName[:validation.DNS1123SubdomainMaxLength]
	}
	return newName
}

// findRelevantEviction finds the most relevant Eviction for the EvictionRequest
// 1. an Eviction with TargetEvicted=True condition
// 2. the oldest active Eviction
// 3. an Eviction with Failed=True condition
// 4. nil if no Eviction matches
func findRelevantEviction(evictions []*lifecyclev1alpha1.Eviction) (*lifecyclev1alpha1.Eviction, bool) {
	var active, succeeded, failed *lifecyclev1alpha1.Eviction

	for _, eviction := range evictions {
		switch {
		case hasEvictionSucceeded(eviction):
			// Succeeded eviction with the latest transition time.
			if succeeded == nil || compareConditionTransitionTime(succeeded, eviction, lifecyclev1alpha1.EvictionConditionTargetEvicted) < 0 {
				succeeded = eviction
			}
		case hasEvictionFailed(eviction):
			// Failed eviction with the latest transition time.
			if failed == nil || compareConditionTransitionTime(failed, eviction, lifecyclev1alpha1.EvictionConditionFailed) < 0 {
				failed = eviction
			}
		default:
			// Oldest active eviction.
			if active == nil || active.CreationTimestamp.Time.After(eviction.CreationTimestamp.Time) {
				active = eviction
			}
		}
	}
	if succeeded != nil {
		return succeeded, false
	}
	if active != nil {
		return active, false
	}
	return failed, true
}

func convertToEvictionRequestConditions(now time.Time, evictionRequest *lifecyclev1alpha1.EvictionRequest, eviction *lifecyclev1alpha1.Eviction, conditionType lifecyclev1alpha1.EvictionConditionType) *metav1ac.ConditionApplyConfiguration {
	if eviction != nil {
		if condition := meta.FindStatusCondition(eviction.Status.Conditions, string(conditionType)); condition != nil {
			return setCondition(now, evictionRequest.Status.Conditions, conditionType,
				condition.Status, lifecyclev1alpha1.EvictionConditionReason(condition.Reason), condition.Message)
		}
	}
	return setCondition(now, evictionRequest.Status.Conditions, conditionType,
		metav1.ConditionFalse, lifecyclev1alpha1.EvictionConditionReasonAwaitingEviction, "waiting for an Eviction to report on progress")
}

func compareConditionTransitionTime(a, b *lifecyclev1alpha1.Eviction, conditionType lifecyclev1alpha1.EvictionConditionType) int {
	aCond := meta.FindStatusCondition(a.Status.Conditions, string(conditionType))
	bCond := meta.FindStatusCondition(b.Status.Conditions, string(conditionType))
	if aCond == nil && bCond == nil {
		return 0
	}
	if aCond == nil && bCond != nil {
		return -1
	}
	if aCond != nil && bCond == nil {
		return 1
	}
	return aCond.LastTransitionTime.Time.Compare(bCond.LastTransitionTime.Time)
}

// setCondition creates or updates a condition in the apply configuration,
// preserving the LastTransitionTime when the status has not changed.
// Inspired by k8s.io/apimachinery/pkg/api/meta.SetStatusCondition.
func setCondition(
	now time.Time,
	existingConditions []metav1.Condition,
	conditionType lifecyclev1alpha1.EvictionConditionType,
	status metav1.ConditionStatus,
	reason lifecyclev1alpha1.EvictionConditionReason,
	message string,
) *metav1ac.ConditionApplyConfiguration {
	transitionTime := metav1.Time{Time: now}

	existing := meta.FindStatusCondition(existingConditions, string(conditionType))
	if existing != nil && existing.Status == status && !existing.LastTransitionTime.IsZero() {
		// Status unchanged, so we preserve the original transition time
		transitionTime = existing.LastTransitionTime
	}

	return metav1ac.Condition().
		WithType(string(conditionType)).
		WithStatus(status).
		WithReason(string(reason)).
		WithMessage(message).
		WithLastTransitionTime(transitionTime)
}

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

// hasCompleted returns true if the Eviction has reached
// a terminal state (Canceled or Evicted condition is True).
func hasEvictionRequestCompleted(evictionRequest *lifecyclev1alpha1.EvictionRequest) bool {
	if meta.IsStatusConditionTrue(evictionRequest.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionTargetEvicted)) {
		return true
	}
	failedCondition := meta.FindStatusCondition(evictionRequest.Status.Conditions, string(lifecyclev1alpha1.EvictionConditionFailed))
	if failedCondition != nil && failedCondition.Status == metav1.ConditionTrue {
		switch failedCondition.Reason {
		case string(lifecyclev1alpha1.EvictionConditionReasonEvictionInvalid):
			return true
		}
	}
	return false
}

func hasEvictionIntent(targetEvictionRequests []*lifecyclev1alpha1.EvictionRequest) bool {
	for _, request := range targetEvictionRequests {
		// only non deleted requests are considered active
		if request.DeletionTimestamp == nil && request.Spec.Intent == lifecyclev1alpha1.EvictionRequestIntentEviction {
			return true
		}
	}
	return false
}

func evictionRequestAsOwnerReference(evictionRequest *lifecyclev1alpha1.EvictionRequest) *metav1ac.OwnerReferenceApplyConfiguration {
	gvk := lifecyclev1alpha1.SchemeGroupVersion.WithKind("EvictionRequest")
	return metav1ac.OwnerReference().
		WithKind(gvk.Kind).
		WithAPIVersion(gvk.GroupVersion().String()).
		WithName(evictionRequest.Name).
		WithUID(evictionRequest.UID)
}

// evictionLabelsNeedSSAUpdate reports label changes that are interesting for us: additions, changes,
// and removal of only a single label.
func evictionLabelsNeedSSAUpdate(oldLabels, newLabels map[string]string) bool {
	for k, v := range newLabels {
		if oldLabels == nil || oldLabels[k] != v {
			// new label added or updated
			return true
		}
	}
	for k, v := range oldLabels {
		switch v {
		// Only requester labels can be removed for now, as responders are immutable.
		case string(lifecyclev1alpha1.EvictionParticipantRoleRequester):
			if len(newLabels[k]) == 0 {
				// requester removed
				return true
			}
		}
	}
	return false
}

// shouldDeferCompletion returns how long to wait before setting the completion
// condition, giving the active responder time to report its final status
// before the eviction is finalized. Returns 0 when no deferral is needed.
func shouldDeferCompletion(now time.Time, activeResponderStatus *lifecyclev1alpha1.ResponderStatus, target targetInfo) *time.Duration {
	if activeResponderStatus == nil || activeResponderStatus.CompletionTime != nil {
		return nil
	}

	meta := target.GetObjectMeta()
	if meta == nil || meta.GetDeletionTimestamp() == nil {
		return nil
	}

	if remaining := GracefulCompletionDelay - now.Sub(meta.GetDeletionTimestamp().Time); remaining > 0 {
		return new(remaining)
	}
	return nil
}

func findTargetResponderIdx(targetResponders []lifecyclev1alpha1.TargetResponder, state lifecyclev1alpha1.ResponderStateType) int {
	for i, responder := range targetResponders {
		if responder.State == state {
			return i
		}
	}
	return -1
}

// findResponderStatus finds the status for a given responder name.
func findResponderStatus(statuses []lifecyclev1alpha1.ResponderStatus, name string) *lifecyclev1alpha1.ResponderStatus {
	for i := range statuses {
		if statuses[i].Name == name {
			return &statuses[i]
		}
	}
	return nil
}

// SortTargetResponders returns highest priority responders on a lower index.
// If there are responders with the same priority, the responder whose domain name comes first in the
// alphabetical higher domain order, will be picked. This means that the top domain labels are compared
// alphabetically first, followed by the lower domain labels. The key is compared last.
//
// NOTE: this is a wrapper around validation.SortTargetResponders in order not to duplicate the sorting logic
func sortTargetResponders(targetResponders []lifecyclev1alpha1.TargetResponder) []lifecyclev1alpha1.TargetResponder {
	if len(targetResponders) == 0 {
		return nil
	}
	input := make([]lifecycle.TargetResponder, 0, len(targetResponders))
	for _, responder := range targetResponders {
		input = append(input, lifecycle.TargetResponder{
			Name:     responder.Name,
			Priority: responder.Priority,
			State:    lifecycle.ResponderStateType(responder.State),
		})
	}
	validation2.SortTargetResponders(input)
	output := make([]lifecyclev1alpha1.TargetResponder, 0, len(targetResponders))
	for _, responder := range input {
		output = append(output, lifecyclev1alpha1.TargetResponder{
			Name:     responder.Name,
			Priority: responder.Priority,
			State:    lifecyclev1alpha1.ResponderStateType(responder.State),
		})
	}
	return output
}
