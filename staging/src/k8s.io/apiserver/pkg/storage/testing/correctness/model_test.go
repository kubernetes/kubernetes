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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
)

func TestCorrectness(t *testing.T) {
	model := NewEmptyModel("")
	steps := correctnessTestSteps()
	executedOps := make([]Operation, 0, len(steps))

	for _, step := range steps {
		t.Run(step.Name, func(t *testing.T) {
			for i, invalidResponse := range step.InvalidResponses {
				ok, _ := model.Step(step.Request, invalidResponse)
				require.False(t, ok, "alternative response #%d should return ok=false: req=%+v resp=%+v", i, step.Request, invalidResponse)
			}

			ok, next := model.Step(step.Request, step.CorrectResponse)
			require.True(t, ok, "valid response should return ok=true: req=%+v resp=%+v", step.Request, step.CorrectResponse)
			model = next
		})
		executedOps = append(executedOps, Operation{
			Request:  step.Request,
			Response: step.CorrectResponse,
		})
	}

	t.Run("WatchCorrectness", func(t *testing.T) {
		versioner := storage.APIObjectVersioner{}
		history := NewWatchHistory(executedOps, versioner)

		var allExpectedEvents []watch.Event
		for _, step := range steps {
			if step.ExpectedEvent != nil {
				allExpectedEvents = append(allExpectedEvents, *step.ExpectedEvent)
			}
		}

		// 1. Validate full watch on /pods/ from RV 1
		allEvents := history.ExpectedEvents("/pods/", 1, storage.Everything)
		require.Equal(t, allExpectedEvents, allEvents)

		ValidateWatchGuarantees(t, versioner, history,
			WatchRequest{
				Name:            "watch-all-simulated",
				Key:             "/pods/",
				ResourceVersion: "1",
			},
			WatchResponse{
				Events: allEvents,
			},
		)

		// 2. Validate filtered watch on /pods/ matching metadata.name=pod1
		predPod1 := storage.SelectionPredicate{Field: fields.OneTermEqualSelector("metadata.name", "pod1")}
		pod1Events := history.ExpectedEvents("/pods/", 1, predPod1)
		var expectedPod1Events []watch.Event
		for _, ev := range allExpectedEvents {
			if ev.Object != nil {
				acc, _ := meta.Accessor(ev.Object)
				if acc.GetName() == "pod1" {
					expectedPod1Events = append(expectedPod1Events, ev)
				}
			}
		}
		require.Equal(t, expectedPod1Events, pod1Events)

		ValidateWatchGuarantees(t, versioner, history,
			WatchRequest{
				Name:            "watch-pod1-simulated",
				Key:             "/pods/",
				ResourceVersion: "1",
				Predicate:       predPod1,
			},
			WatchResponse{
				Events: pod1Events,
			},
		)

		// 3. Validate resumed watch from RV 2
		from2Events := history.ExpectedEvents("/pods/", 2, storage.Everything)
		var expectedFrom2Events []watch.Event
		for _, ev := range allExpectedEvents {
			if ev.Object != nil {
				acc, _ := meta.Accessor(ev.Object)
				rv, _ := versioner.ParseResourceVersion(acc.GetResourceVersion())
				if rv > 2 {
					expectedFrom2Events = append(expectedFrom2Events, ev)
				}
			}
		}
		require.Equal(t, expectedFrom2Events, from2Events)

		ValidateWatchGuarantees(t, versioner, history,
			WatchRequest{
				Name:            "watch-from2-simulated",
				Key:             "/pods/",
				ResourceVersion: "2",
			},
			WatchResponse{
				Events: from2Events,
			},
		)
	})
}
