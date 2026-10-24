/*
Copyright 2021 The Kubernetes Authors.

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

package job

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

// matchPodFailurePolicy returns information about matching a given failed pod
// against the pod failure policy rules. The information is represented as an
//   - optional job failure message (present in case the pod matched a 'FailJob' rule),
//   - a boolean indicating if the failure should be counted towards backoffLimit
//     (and backoffLimitPerIndex if specified). It should not be counted
//     if the pod matched an 'Ignore' rule,
//   - a pointer to the matched pod failure policy action.
func matchPodFailurePolicy(podFailurePolicy *batch.PodFailurePolicy, failedPod *v1.Pod) (*string, bool, *batch.PodFailurePolicyAction) {
	msg, countFailed, ruleIndex := matchPodFailurePolicyRule(podFailurePolicy, failedPod)
	if ruleIndex < 0 {
		return msg, countFailed, nil
	}
	action := podFailurePolicy.Rules[ruleIndex].Action
	return msg, countFailed, &action
}

// matchPodFailurePolicyRule is like matchPodFailurePolicy, but returns the
// index of the matched rule instead of its action, or -1 if no rule matched.
func matchPodFailurePolicyRule(podFailurePolicy *batch.PodFailurePolicy, failedPod *v1.Pod) (*string, bool, int) {
	if podFailurePolicy == nil {
		return nil, true, -1
	}
	for index, podFailurePolicyRule := range podFailurePolicy.Rules {
		if podFailurePolicyRule.OnExitCodes != nil {
			if containerStatus := matchOnExitCodes(&failedPod.Status, podFailurePolicyRule.OnExitCodes); containerStatus != nil {
				switch podFailurePolicyRule.Action {
				case batch.PodFailurePolicyActionIgnore:
					return nil, false, index
				case batch.PodFailurePolicyActionFailIndex, batch.PodFailurePolicyActionCount:
					return nil, true, index
				case batch.PodFailurePolicyActionFailJob:
					msg := fmt.Sprintf("Container %s for pod %s/%s failed with exit code %v matching %v rule at index %d",
						containerStatus.Name, failedPod.Namespace, failedPod.Name, containerStatus.State.Terminated.ExitCode, podFailurePolicyRule.Action, index)
					return &msg, true, index
				}
			}
		} else if podFailurePolicyRule.OnPodConditions != nil {
			if podCondition := matchOnPodConditions(&failedPod.Status, podFailurePolicyRule.OnPodConditions); podCondition != nil {
				switch podFailurePolicyRule.Action {
				case batch.PodFailurePolicyActionIgnore:
					return nil, false, index
				case batch.PodFailurePolicyActionFailIndex, batch.PodFailurePolicyActionCount:
					return nil, true, index
				case batch.PodFailurePolicyActionFailJob:
					msg := fmt.Sprintf("Pod %s/%s has condition %v matching %v rule at index %d",
						failedPod.Namespace, failedPod.Name, podCondition.Type, podFailurePolicyRule.Action, index)
					return &msg, true, index
				}
			}
		}
	}
	return nil, true, -1
}

// failJobReason returns the Job failure condition reason for the FailJob rule
// at ruleIndex: "PodFailurePolicy_<name>", or "PodFailurePolicy_<index>" for an
// unnamed rule. It returns the legacy "PodFailurePolicy" reason while the
// PodFailurePolicyName feature gate is disabled.
func failJobReason(podFailurePolicy *batch.PodFailurePolicy, ruleIndex int) string {
	if !feature.DefaultFeatureGate.Enabled(features.PodFailurePolicyName) {
		return batch.JobReasonPodFailurePolicy
	}
	if name := podFailurePolicy.Rules[ruleIndex].Name; name != nil {
		return batch.JobReasonPodFailurePolicy + "_" + *name
	}
	return batch.JobReasonPodFailurePolicy + "_" + strconv.Itoa(ruleIndex)
}

// metricsReason maps a Job finished condition reason to the bounded set of
// values used for the jobs_finished_total reason label.
func metricsReason(reason string) string {
	if strings.HasPrefix(reason, batch.JobReasonPodFailurePolicy+"_") {
		return batch.JobReasonPodFailurePolicy
	}
	return reason
}

// matchOnExitCodes returns a terminated container status that matches the error code requirement, if any exists.
// If the returned status is non-nil, it has a non-nil Terminated field.
func matchOnExitCodes(podStatus *v1.PodStatus, requirement *batch.PodFailurePolicyOnExitCodesRequirement) *v1.ContainerStatus {
	if containerStatus := getMatchingContainerFromList(podStatus.ContainerStatuses, requirement); containerStatus != nil {
		return containerStatus
	}
	return getMatchingContainerFromList(podStatus.InitContainerStatuses, requirement)
}

func matchOnPodConditions(podStatus *v1.PodStatus, requirement []batch.PodFailurePolicyOnPodConditionsPattern) *v1.PodCondition {
	for _, podCondition := range podStatus.Conditions {
		for _, pattern := range requirement {
			if podCondition.Type == pattern.Type && podCondition.Status == pattern.Status {
				return &podCondition
			}
		}
	}
	return nil
}

// getMatchingContainerFromList returns the first terminated container status in the list that matches the error code requirement, or nil if none match.
// If the returned status is non-nil, it has a non-nil Terminated field
func getMatchingContainerFromList(containerStatuses []v1.ContainerStatus, requirement *batch.PodFailurePolicyOnExitCodesRequirement) *v1.ContainerStatus {
	for _, containerStatus := range containerStatuses {
		if containerStatus.State.Terminated == nil {
			// This container is still be terminating. There is no exit code to match.
			continue
		}
		if requirement.ContainerName == nil || *requirement.ContainerName == containerStatus.Name {
			if containerStatus.State.Terminated.ExitCode != 0 {
				if isOnExitCodesOperatorMatching(containerStatus.State.Terminated.ExitCode, requirement) {
					return &containerStatus
				}
			}
		}
	}
	return nil
}

func isOnExitCodesOperatorMatching(exitCode int32, requirement *batch.PodFailurePolicyOnExitCodesRequirement) bool {
	switch requirement.Operator {
	case batch.PodFailurePolicyOnExitCodesOpIn:
		return slices.Contains(requirement.Values, exitCode)
	case batch.PodFailurePolicyOnExitCodesOpNotIn:
		return !slices.Contains(requirement.Values, exitCode)
	default:
		return false
	}
}
