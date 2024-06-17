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

package job

import (
	"fmt"

	batch "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

func matchSuccessPolicy(logger klog.Logger, successPolicy *batch.SuccessPolicy, completions int32, succeededIndexes orderedIntervals) (string, bool) {
	if !feature.DefaultFeatureGate.Enabled(features.JobSuccessPolicy) || successPolicy == nil || len(succeededIndexes) == 0 {
		return "", false
	}

	rulesMatchedMsg := "Matched rules at index"
	for index, rule := range successPolicy.Rules {
		if rule.SucceededIndexes != nil {
			requiredIndexes := parseIndexesFromString(logger, *rule.SucceededIndexes, int(completions))
			// Failed to parse succeededIndexes of the rule due to some errors like invalid format.
			if len(requiredIndexes) == 0 {
				continue
			}
			if matchSucceededIndexesRule(requiredIndexes, succeededIndexes, rule.SucceededCount) {
				return fmt.Sprintf("%s %d", rulesMatchedMsg, index), true
			}
		} else if rule.SucceededCount != nil && succeededIndexes.total() >= int(*rule.SucceededCount) {
			return fmt.Sprintf("%s %d", rulesMatchedMsg, index), true
		}
	}
	return "", false
}

func hasSuccessCriteriaMetCondition(job *batch.Job) *batch.JobCondition {
	if feature.DefaultFeatureGate.Enabled(features.JobSuccessPolicy) {
		successCriteriaMet := findConditionByType(job.Status.Conditions, batch.JobSuccessCriteriaMet)
		if successCriteriaMet != nil && successCriteriaMet.Status == v1.ConditionTrue {
			return successCriteriaMet
		}
	}
	return nil
}

func isSuccessCriteriaMetCondition(cond *batch.JobCondition) bool {
	return feature.DefaultFeatureGate.Enabled(features.JobSuccessPolicy) &&
		cond != nil && cond.Type == batch.JobSuccessCriteriaMet && cond.Status == v1.ConditionTrue
}

func matchSucceededIndexesRule(ruleIndexes, succeededIndexes orderedIntervals, succeededCount *int32) bool {
	var contains, succeededPointer, rulePointer int
	for rulePointer < len(ruleIndexes) && succeededPointer < len(succeededIndexes) {
		if overlap := min(ruleIndexes[rulePointer].Last, succeededIndexes[succeededPointer].Last) -
			max(ruleIndexes[rulePointer].First, succeededIndexes[succeededPointer].First) + 1; overlap > 0 {
			contains += overlap
		}
		if succeededIndexes[succeededPointer].Last < ruleIndexes[rulePointer].Last {
			// The current succeeded interval is behind, so we can move to the next.
			succeededPointer++
		} else if succeededIndexes[succeededPointer].Last > ruleIndexes[rulePointer].Last {
			// The current rule interval is behind, so we can move to the next.
			rulePointer++
		} else {
			// Both intervals end at the same position, we can move to the next succeeded, and next rule.
			succeededPointer++
			rulePointer++
		}
	}
	return contains == ruleIndexes.total() || (succeededCount != nil && contains >= int(*succeededCount))
}
