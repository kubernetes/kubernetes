/*
Copyright 2019 The Kubernetes Authors.

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

package apihelpers

import (
	"sort"

	flowcontrol "k8s.io/api/flowcontrol/v1beta3"
)

// SetFlowSchemaCondition sets conditions.
func SetFlowSchemaCondition(flowSchema *flowcontrol.FlowSchema, newCondition flowcontrol.FlowSchemaCondition) {
	existingCondition := GetFlowSchemaConditionByType(flowSchema, newCondition.Type)
	if existingCondition == nil {
		flowSchema.Status.Conditions = append(flowSchema.Status.Conditions, newCondition)
		return
	}

	if existingCondition.Status != newCondition.Status {
		existingCondition.Status = newCondition.Status
		existingCondition.LastTransitionTime = newCondition.LastTransitionTime
	}

	existingCondition.Reason = newCondition.Reason
	existingCondition.Message = newCondition.Message
}

// GetFlowSchemaConditionByType gets conditions.
func GetFlowSchemaConditionByType(flowSchema *flowcontrol.FlowSchema, conditionType flowcontrol.FlowSchemaConditionType) *flowcontrol.FlowSchemaCondition {
	for i := range flowSchema.Status.Conditions {
		if flowSchema.Status.Conditions[i].Type == conditionType {
			return &flowSchema.Status.Conditions[i]
		}
	}
	return nil
}

// SetPriorityLevelConfigurationCondition sets conditions.
func SetPriorityLevelConfigurationCondition(priorityLevel *flowcontrol.PriorityLevelConfiguration, newCondition flowcontrol.PriorityLevelConfigurationCondition) {
	existingCondition := GetPriorityLevelConfigurationConditionByType(priorityLevel, newCondition.Type)
	if existingCondition == nil {
		priorityLevel.Status.Conditions = append(priorityLevel.Status.Conditions, newCondition)
		return
	}

	if existingCondition.Status != newCondition.Status {
		existingCondition.Status = newCondition.Status
		existingCondition.LastTransitionTime = newCondition.LastTransitionTime
	}

	existingCondition.Reason = newCondition.Reason
	existingCondition.Message = newCondition.Message
}

// GetPriorityLevelConfigurationConditionByType gets conditions.
func GetPriorityLevelConfigurationConditionByType(priorityLevel *flowcontrol.PriorityLevelConfiguration, conditionType flowcontrol.PriorityLevelConfigurationConditionType) *flowcontrol.PriorityLevelConfigurationCondition {
	for i := range priorityLevel.Status.Conditions {
		if priorityLevel.Status.Conditions[i].Type == conditionType {
			return &priorityLevel.Status.Conditions[i]
		}
	}
	return nil
}

var _ sort.Interface = FlowSchemaSequence{}

// FlowSchemaSequence holds sorted set of pointers to FlowSchema objects.
// FlowSchemaSequence implements `sort.Interface`
type FlowSchemaSequence []*flowcontrol.FlowSchema

func (s FlowSchemaSequence) Len() int {
	return len(s)
}

func (s FlowSchemaSequence) Less(i, j int) bool {
	// the flow-schema w/ lower matching-precedence is prior
	if ip, jp := s[i].Spec.MatchingPrecedence, s[j].Spec.MatchingPrecedence; ip != jp {
		return ip < jp
	}
	// sort alphabetically
	return s[i].Name < s[j].Name
}

func (s FlowSchemaSequence) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
