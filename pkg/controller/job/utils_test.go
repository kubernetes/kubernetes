/*
Copyright 2016 The Kubernetes Authors.

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
	"testing"

	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"reflect"
	"strconv"
)

func TestIsJobFinished(t *testing.T) {
	testCases := map[string]struct {
		conditionType        batch.JobConditionType
		conditionStatus      v1.ConditionStatus
		expectJobNotFinished bool
	}{
		"Job is completed and condition is true": {
			batch.JobComplete,
			v1.ConditionTrue,
			false,
		},
		"Job is completed and condition is false": {
			batch.JobComplete,
			v1.ConditionFalse,
			true,
		},
		"Job is completed and condition is unknown": {
			batch.JobComplete,
			v1.ConditionUnknown,
			true,
		},
		"Job is failed and condition is true": {
			batch.JobFailed,
			v1.ConditionTrue,
			false,
		},
		"Job is failed and condition is false": {
			batch.JobFailed,
			v1.ConditionFalse,
			true,
		},
		"Job is failed and condition is unknown": {
			batch.JobFailed,
			v1.ConditionUnknown,
			true,
		},
	}

	for name, tc := range testCases {
		job := &batch.Job{
			Status: batch.JobStatus{
				Conditions: []batch.JobCondition{{
					Type:   tc.conditionType,
					Status: tc.conditionStatus,
				}},
			},
		}

		if tc.expectJobNotFinished == IsJobFinished(job) {
			if tc.expectJobNotFinished {
				t.Errorf("test name: %s, job was not expected to be finished", name)
			} else {
				t.Errorf("test name: %s, job was expected to be finished", name)
			}
		}
	}
}

func TestGetCompletionsIndex(t *testing.T) {
	testCases := map[string]struct {
		pod    *v1.Pod
		result int
		error  bool
	}{
		"Pod has index 1": {
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						CompletionsIndexName: "1",
					},
				},
			},
			1,
			false,
		},
		"Pod has index 5555": {
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						CompletionsIndexName: "5555",
					},
				},
			},
			5555,
			false,
		},
		"Pod has index -1": {
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						CompletionsIndexName: "-1",
					},
				},
			},
			-1,
			false,
		},
		"Pod has err": {
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						CompletionsIndexName: "a",
					},
				},
			},
			1,
			true,
		},
	}

	for name, tc := range testCases {
		result, err := getCompletionsIndex(tc.pod)
		if tc.error {
			if err == nil {
				t.Errorf("test name: %s, get completions index should return error", name)
			}
			continue
		}
		if result != tc.result {
			t.Errorf("test name: %s, result should be %d, but %d", name, tc.result, result)
		}
	}
}

func TestAddCompletionsIndexToPodTemplate(t *testing.T) {
	job := &batch.Job{
		Spec: batch.JobSpec{
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{},
				},
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Env: []v1.EnvVar{},
						},
					},
					Containers: []v1.Container{
						{
							Env: []v1.EnvVar{},
						},
						{
							Env: []v1.EnvVar{},
						},
					},
				},
			},
		},
	}
	testCases := map[string]struct {
		job                    *batch.Job
		completionsIndex       int32
		shouldEq               bool
		shouldCompletionsIndex int
	}{
		"template add index 1": {
			job,
			1,
			false,
			1,
		},
		"template add index 0": {
			job,
			0,
			true,
			-1,
		},
		"template add index -1": {
			job,
			-1,
			true,
			-1,
		},
	}

	for name, tc := range testCases {
		result := addCompletionsIndexToPodTemplate(tc.job, tc.completionsIndex)
		if tc.shouldEq {
			if !reflect.DeepEqual(result, job.Spec.Template) {
				t.Errorf("test name: %s, resut should eq job.Spec.Template", name)
			}
			continue
		}
		if result.Labels[CompletionsIndexName] != strconv.Itoa(tc.shouldCompletionsIndex) {
			t.Errorf("test name: %s, labels CompletionsIndexName should be %d, but %s", name, tc.shouldCompletionsIndex, result.Labels[CompletionsIndexName])
			continue
		}
		containers := append(result.Spec.Containers, result.Spec.InitContainers...)
		for _, c := range containers {
			hasEnv := false
			for _, env := range c.Env {
				if env.Name == CompletionsIndexEnvArgName {
					hasEnv = true
					if env.Value != strconv.Itoa(tc.shouldCompletionsIndex) {
						t.Errorf("test name: %s, env CompletionsIndexEnvArgName should be %d, but %s", name, tc.shouldCompletionsIndex, env.Value)
					}
				}
			}
			if !hasEnv {
				t.Errorf("test name: %s, env CompletionsIndexEnvArgName must have", name)
			}
		}
	}
}

func TestDistinctPods(t *testing.T) {
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "1",
			},
		},
	}
	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "2",
			},
		},
	}
	pod3 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "3",
			},
		},
	}
	testCases := map[string]struct {
		pods   []*v1.Pod
		should int32
	}{
		"distinct should 3": {
			[]*v1.Pod{
				pod1,
				pod2,
				pod3,
			},
			3,
		},
		"distinct should 3 , len 4": {
			[]*v1.Pod{
				pod1,
				pod2,
				pod3,
				pod3,
			},
			3,
		},
		"distinct should 2, len 4": {
			[]*v1.Pod{
				pod1,
				pod3,
				pod3,
				pod1,
			},
			2,
		},
	}
	for name, tc := range testCases {
		result := distinctPods(tc.pods)
		if result != tc.should {
			t.Errorf("test name: %s, distinct should be %d, but %d", name, tc.should, result)
		}
	}
}

func TestGetAvailableCompletionsIndexes(t *testing.T) {
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "1",
			},
		},
	}
	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "2",
			},
		},
	}
	pod3 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "3",
			},
		},
	}
	pod4 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "4",
			},
		},
	}
	testCases := map[string]struct {
		activePods    []*v1.Pod
		succeededPods []*v1.Pod
		completions   int32
		result        []int32
	}{
		"available is 1": {
			[]*v1.Pod{
				pod2,
				pod3,
			},
			[]*v1.Pod{
				pod2,
				pod4,
			},
			4,
			[]int32{
				1,
			},
		},
		"available is 1,2": {
			[]*v1.Pod{
				pod3,
			},
			[]*v1.Pod{
				pod3,
				pod4,
			},
			4,
			[]int32{
				1,
				2,
			},
		},
		"available is empty": {
			[]*v1.Pod{
				pod1,
				pod2,
				pod3,
			},
			[]*v1.Pod{
				pod1,
				pod2,
				pod3,
			},
			3,
			[]int32{},
		},
	}
	for name, tc := range testCases {
		result := getAvailableCompletionsIndexes(tc.activePods, tc.succeededPods, tc.completions)
		if !reflect.DeepEqual(result, tc.result) {
			t.Errorf("test name: %s, result should be %+v, but %+v", name, tc.result, result)
		}
	}
}

func TestGetNeedStopActivePods(t *testing.T) {
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "1",
			},
		},
	}
	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "2",
			},
		},
	}
	pod3 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "3",
			},
		},
	}
	pod4 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "4",
			},
		},
	}
	testCases := map[string]struct {
		activePods    []*v1.Pod
		succeededPods []*v1.Pod
		result        []*v1.Pod
	}{
		"need stop is pod2": {
			[]*v1.Pod{
				pod2,
				pod3,
			},
			[]*v1.Pod{
				pod2,
				pod4,
			},
			[]*v1.Pod{
				pod2,
			},
		},
		"need stop is pod1, pod4": {
			[]*v1.Pod{
				pod1,
				pod2,
				pod3,
				pod4,
				pod4,
			},
			[]*v1.Pod{
				pod1,
			},
			[]*v1.Pod{
				pod1,
				pod4,
			},
		},
		"need stop is pod3": {
			[]*v1.Pod{
				pod2,
				pod3,
				pod3,
			},
			[]*v1.Pod{
				pod4,
			},
			[]*v1.Pod{
				pod3,
			},
		},
	}
	for name, tc := range testCases {
		result := getNeedStopActivePods(tc.activePods, tc.succeededPods)
		if !reflect.DeepEqual(result, tc.result) {
			t.Errorf("test name: %s, result should be %+v, but %+v", name, tc.result, result)
		}
	}
}

func TestGetCompletionsIndexesAndDuplicateIndexPods(t *testing.T) {
	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "1",
			},
		},
	}
	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "2",
			},
		},
	}
	pod2x := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "2",
				"test":               "test",
			},
		},
	}
	pod3 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "3",
			},
		},
	}
	pod4 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				CompletionsIndexName: "4",
			},
		},
	}
	testCases := map[string]struct {
		pods               []*v1.Pod
		hasIndexes         map[int]bool
		duplicateIndexPods []*v1.Pod
	}{
		"pods is 2,3": {
			[]*v1.Pod{
				pod2,
				pod3,
				pod1,
			},
			map[int]bool{
				2: true,
				3: true,
				1: true,
			},
			[]*v1.Pod{},
		},
		"pods is 2,3,2,4": {
			[]*v1.Pod{
				pod2,
				pod3,
				pod2x,
				pod4,
			},
			map[int]bool{
				2: true,
				3: true,
				4: true,
			},
			[]*v1.Pod{
				pod2x,
			},
		},
		"pods is 2,3,2,2,4": {
			[]*v1.Pod{
				pod2,
				pod3,
				pod2x,
				pod2,
				pod4,
			},
			map[int]bool{
				2: true,
				3: true,
				4: true,
			},
			[]*v1.Pod{
				pod2x,
				pod2,
			},
		},
	}
	for name, tc := range testCases {
		indexes, dupPods := getCompletionsIndexesAndDuplicateIndexPods(tc.pods)
		bool1 := reflect.DeepEqual(indexes, tc.hasIndexes)
		bool2 := reflect.DeepEqual(dupPods, tc.duplicateIndexPods)
		if !(bool1 && bool2) {
			t.Errorf("test name: %s, result should be indexes: %+v; dupPods: %+v, but indexes: %+v; dupPods: %+v", name, tc.hasIndexes, tc.duplicateIndexPods, indexes, dupPods)
		}
	}
}
