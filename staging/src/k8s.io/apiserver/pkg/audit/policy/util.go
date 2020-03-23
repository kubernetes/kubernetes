/*
Copyright 2018 The Kubernetes Authors.

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

package policy

import (
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/apis/audit"
)

// AllStages returns all possible stages
func AllStages() sets.String {
	return sets.NewString(
		audit.StageRequestReceived,
		audit.StageResponseStarted,
		audit.StageResponseComplete,
		audit.StagePanic,
	)
}

// AllLevels returns all possible levels
func AllLevels() sets.String {
	return sets.NewString(
		string(audit.LevelNone),
		string(audit.LevelMetadata),
		string(audit.LevelRequest),
		string(audit.LevelRequestResponse),
	)
}

// InvertStages subtracts the given array of stages from all stages
func InvertStages(stages []audit.Stage) []audit.Stage {
	s := ConvertStagesToStrings(stages)
	a := AllStages()
	a.Delete(s...)
	return ConvertStringSetToStages(a)
}

// ConvertStagesToStrings converts an array of stages to a string array
func ConvertStagesToStrings(stages []audit.Stage) []string {
	s := make([]string, len(stages))
	for i, stage := range stages {
		s[i] = string(stage)
	}
	return s
}

// ConvertStringSetToStages converts a string set to an array of stages
func ConvertStringSetToStages(set sets.String) []audit.Stage {
	stages := make([]audit.Stage, len(set))
	for i, stage := range set.List() {
		stages[i] = audit.Stage(stage)
	}
	return stages
}
