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

package correctness

import (
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	"k8s.io/apiserver/pkg/storage"
)

func TestCorrectness(t *testing.T) {
	versioner := &storage.APIObjectVersioner{}
	newPod := func() runtime.Object { return &example.Pod{} }
	initialState := NewEmptyModel("", newPod, func() runtime.Object { return &example.PodList{} }, versioner)
	model := initialState.Clone()
	steps := correctnessTestSteps()
	history := make([]Operation, len(steps))
	for i, step := range steps {
		history[i] = Operation{Request: step.Request, Response: step.CorrectResponse}
	}

	var expectEvents, gotEvents []watch.Event
	for _, step := range steps {
		t.Run(step.Name, func(t *testing.T) {
			if step.ExpectedEvent != nil {
				expectEvents = append(expectEvents, *step.ExpectedEvent)
			}

			for i, invalidResponse := range step.InvalidResponses {
				ok, _, _ := model.Step(step.Request, invalidResponse)
				require.False(t, ok, "alternative response #%d should fail validation: req=%+v resp=%+v", i, step.Request, invalidResponse)
			}

			ok, next, change := model.Step(step.Request, step.CorrectResponse)
			require.True(t, ok, "valid response should return ok=true: req=%+v resp=%+v", step.Request, step.CorrectResponse)
			model = next
			if change != nil {
				event, err := change.toWatchEvent(storage.APIObjectVersioner{}, storage.Everything)
				require.NoError(t, err)
				gotEvents = append(gotEvents, *event)
			}
		})
	}
	require.Equal(t, expectEvents, gotEvents)

	replay, err := NewReplay(initialState, history)
	require.NoError(t, err)

	for _, tc := range readTestCases() {
		t.Run(tc.Name, func(t *testing.T) {
			for i, invalidResponse := range tc.InvalidResponses {
				ok, _, _ := model.Step(tc.Request, invalidResponse)
				if ok {
					err := replay.Validate(tc.Request, invalidResponse)
					require.Error(t, err, "alternative response #%d should fail validation: req=%+v resp=%+v", i, tc.Request, invalidResponse)
				}
			}

			ok, next, change := model.Step(tc.Request, tc.CorrectResponse)
			require.True(t, ok, "valid response should return ok=true: req=%+v resp=%+v", tc.Request, tc.CorrectResponse)
			require.Equal(t, model, next)
			require.Nil(t, change)
			require.NoError(t, replay.Validate(tc.Request, tc.CorrectResponse))
		})
	}

	validator := NewWatchValidator(versioner, replay, getKey)
	for _, tc := range watchTestCases() {
		t.Run(tc.Name, func(t *testing.T) {
			if tc.ExpectError != nil {
				require.NoError(t, validator.ValidateWatch(tc.Request, WatchResponse{Err: tc.ExpectError}))
				return
			}
			events, err := replay.Watch(tc.Request)
			require.NoError(t, err)
			require.NoError(t, validator.ValidateWatch(tc.Request, WatchResponse{Events: events}))
		})
	}
}
