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

package storage

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anishathalye/porcupine"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/testing/correctness"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var (
	requestDistribution = []ChoiceWeight[RequestType]{{
		Choice: RequestTypeCreate,
		Weight: 15,
	}, {
		Choice: RequestTypeDelete,
		Weight: 10,
	}, {
		Choice: RequestTypeDeleteUIDPrecondition,
		Weight: 10,
	}, {
		Choice: RequestTypeGet,
		Weight: 25,
	}, {
		Choice: RequestTypeUpdate,
		Weight: 20,
	}, {
		Choice: RequestTypeUpdateUIDPrecondition,
		Weight: 10,
	}, {
		Choice: RequestTypeUpdateNoOp,
		Weight: 5,
	}, {
		Choice: RequestTypeUpdateWithCachedObject,
		Weight: 5,
	}}

	unaryCfg = UnaryConfig{
		Concurrency:         8,
		Namespaces:          2,
		Objects:             4,
		MaxOperations:       10000,
		RequestDistribution: requestDistribution,
	}

	watchRequestDistribution = []ChoiceWeight[WatchRequestType]{{
		Choice: RVEmpty,
		Weight: 15,
	}, {
		Choice: RVZero,
		Weight: 15,
	}, {
		Choice: RVOne,
		Weight: 10,
	}, {
		Choice: RVCurrent,
		Weight: 20,
	}, {
		Choice: RVPast,
		Weight: 20,
	}, {
		Choice: RVFuture,
		Weight: 20,
	}}

	watchCfg = WatchConfig{
		Concurrency:         4,
		Duration:            500 * time.Millisecond,
		MaxEvents:           50,
		RequestDistribution: watchRequestDistribution,
	}
)

func TestCorrectness(t *testing.T) {
	storages := []struct {
		name string
		fn   func(t *testing.T) (storage.Interface, string)
	}{
		{"etcd3", func(t *testing.T) (storage.Interface, string) {
			return setupStore(t, generic.UndecoratedStorage)
		}},
		{"cacher", func(t *testing.T) (storage.Interface, string) {
			return setupStore(t, genericregistry.StorageWithCacher())
		}},
	}
	for _, s := range storages {
		t.Run(s.name, func(t *testing.T) {
			store, storagePrefix := s.fn(t)
			testCorrectness(t, store, storagePrefix)
		})
	}
}

func testCorrectness(t *testing.T, store storage.Interface, storagePrefix string) {
	ctx := t.Context()

	list := &api.PodList{}
	err := store.GetList(ctx, "/pods", storage.ListOptions{Recursive: true, Predicate: storage.Everything}, list)
	require.NoError(t, err)
	initialState, err := correctness.NewModelFromStorage(storagePrefix, list, func() runtime.Object { return &api.Pod{} }, cacheKeyFunc)
	require.NoError(t, err)

	stopWatches := make(chan struct{})
	var operations []correctness.Operation
	var watches []correctness.WatchOperation
	var wg sync.WaitGroup
	wg.Go(func() {
		operations, err = RunUnaryTraffic(ctx, store, unaryCfg)
		close(stopWatches)
	})
	wg.Go(func() {
		watches = RunWatchTraffic(ctx, store, watchCfg, stopWatches)
	})
	wg.Wait()
	require.NoError(t, err)
	watchEvents := 0
	for _, w := range watches {
		watchEvents += len(w.Response.Events)
	}
	t.Logf("Collected %d unary operations and %d watches with %d events",
		len(operations), len(watches), watchEvents)

	model := ToPorcupineModel(initialState)
	res, info := porcupine.CheckOperationsVerbose(model, toPorcupineOperations(operations), time.Minute)

	if artifacts := os.Getenv("ARTIFACTS"); artifacts != "" {
		testName := strings.ReplaceAll(t.Name(), "/", "_")
		path := filepath.Join(artifacts, fmt.Sprintf("%s_linearization_visualization.html", testName))
		if err := porcupine.VisualizePath(model, info, path); err != nil {
			t.Logf("Failed to write linearization visualization: %v", err)
		} else {
			t.Logf("Linearization visualization written to: %s", path)
		}
	}

	require.Equal(t, porcupine.Ok, res, "Linearizability check failed across %d operations", len(operations))
	t.Logf("Linearizability check succeeded across %d operations", len(operations))

	// The model defines no Partition, so an Ok result carries a single
	// complete linearization.
	linearizations := info.PartialLinearizations()
	require.Len(t, linearizations, 1, "expected one partition")
	require.Len(t, linearizations[0], 1, "expected one linearization")
	expectEvents := eventsFromLinearization(initialState, operations, linearizations[0][0])
	validator := correctness.NewWatchValidator(store.Versioner(), cacheKeyFunc, expectEvents)
	for _, w := range watches {
		require.NoError(t, validator.ValidateWatch(w.Request, w.Response))
	}
	require.Positive(t, watchEvents, "expected at least one watch event across %d watches", len(watches))
}

func eventsFromLinearization(initialState *correctness.Model, ops []correctness.Operation, linearization []int) []watch.Event {
	state := initialState.Clone()
	var events []watch.Event
	for _, i := range linearization {
		op := ops[i]
		ok, next, event := state.Step(op.Request, op.Response)
		if !ok {
			panic(fmt.Sprintf("linearized operation %d failed model step", i))
		}
		state = next
		if event != nil {
			events = append(events, *event)
		}
	}
	return events
}

// ToPorcupineModel maps a correctness.Model to porcupine.Model with an initial state.
func ToPorcupineModel(s *correctness.Model) porcupine.Model {
	return porcupine.Model{
		Init: func() any {
			return s.Clone()
		},
		Step: func(state, input, output any) (bool, any) {
			ok, next, _ := state.(*correctness.Model).Step(input.(correctness.Request), output.(correctness.Response))
			return ok, next
		},
		Equal: func(state1, state2 any) bool {
			return state1.(*correctness.Model).Equal(state2.(*correctness.Model))
		},
		DescribeOperation: func(input, output any) string {
			return input.(correctness.Request).Describe(output.(correctness.Response))
		},
		DescribeState: func(state any) string {
			return state.(*correctness.Model).Describe()
		},
	}
}

func toPorcupineOperations(ops []correctness.Operation) []porcupine.Operation {
	pOps := make([]porcupine.Operation, len(ops))
	for i, op := range ops {
		pOps[i] = porcupine.Operation{
			ClientId: int(op.ClientID),
			Input:    op.Request,
			Call:     op.Start.UnixNano(),
			Output:   op.Response,
			Return:   op.End.UnixNano(),
		}
	}
	return pOps
}
