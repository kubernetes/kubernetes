/*
Copyright 2015 The Kubernetes Authors.

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

package scheduler

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/api"
)

func TestAlgorithmNameValidation(t *testing.T) {
	algorithmNamesShouldValidate := []string{
		"1SomeAlgo1rithm",
		"someAlgor-ithm1",
	}
	algorithmNamesShouldNotValidate := []string{
		"-SomeAlgorithm",
		"SomeAlgorithm-",
		"Some,Alg:orithm",
	}
	for _, name := range algorithmNamesShouldValidate {
		t.Run(name, func(t *testing.T) {
			if !validName.MatchString(name) {
				t.Errorf("should be a valid algorithm name but is not valid.")
			}
		})
	}
	for _, name := range algorithmNamesShouldNotValidate {
		t.Run(name, func(t *testing.T) {
			if validName.MatchString(name) {
				t.Errorf("should be an invalid algorithm name but is valid.")
			}
		})
	}
}

func TestBuildScoringFunctionShapeFromRequestedToCapacityRatioArguments(t *testing.T) {
	arguments := api.RequestedToCapacityRatioArguments{
		UtilizationShape: []api.UtilizationShapePoint{
			{Utilization: 10, Score: 1},
			{Utilization: 30, Score: 5},
			{Utilization: 70, Score: 2},
		},
		Resources: []api.ResourceSpec{
			{Name: v1.ResourceCPU},
			{Name: v1.ResourceMemory},
		},
	}
	builtShape, resources := buildScoringFunctionShapeFromRequestedToCapacityRatioArguments(&arguments)
	expectedShape, _ := priorities.NewFunctionShape([]priorities.FunctionShapePoint{
		{Utilization: 10, Score: 10},
		{Utilization: 30, Score: 50},
		{Utilization: 70, Score: 20},
	})
	expectedResources := priorities.ResourceToWeightMap{
		v1.ResourceCPU:    1,
		v1.ResourceMemory: 1,
	}
	assert.Equal(t, expectedShape, builtShape)
	assert.Equal(t, expectedResources, resources)
}

func TestBuildScoringFunctionShapeFromRequestedToCapacityRatioArgumentsNilResourceToWeightMap(t *testing.T) {
	arguments := api.RequestedToCapacityRatioArguments{
		UtilizationShape: []api.UtilizationShapePoint{
			{Utilization: 10, Score: 1},
			{Utilization: 30, Score: 5},
			{Utilization: 70, Score: 2},
		},
	}
	builtShape, resources := buildScoringFunctionShapeFromRequestedToCapacityRatioArguments(&arguments)
	expectedShape, _ := priorities.NewFunctionShape([]priorities.FunctionShapePoint{
		{Utilization: 10, Score: 10},
		{Utilization: 30, Score: 50},
		{Utilization: 70, Score: 20},
	})
	expectedResources := priorities.ResourceToWeightMap{
		v1.ResourceCPU:    1,
		v1.ResourceMemory: 1,
	}
	assert.Equal(t, expectedShape, builtShape)
	assert.Equal(t, expectedResources, resources)
}
