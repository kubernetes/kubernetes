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
	model := NewEmptyModel("", func() runtime.Object { return &example.Pod{} }, func() runtime.Object { return &example.PodList{} }, versioner)

	var expectEvents, gotEvents []watch.Event
	for _, step := range correctnessTestSteps() {
		t.Run(step.Name, func(t *testing.T) {
			if step.ExpectedEvent != nil {
				expectEvents = append(expectEvents, *step.ExpectedEvent)
			}

			for i, invalidResponse := range step.InvalidResponses {
				ok, _, _ := model.Step(step.Request, invalidResponse)
				require.False(t, ok, "alternative response #%d should return ok=false: req=%+v resp=%+v", i, step.Request, invalidResponse)
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
}
