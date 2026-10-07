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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/testing/correctness"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var (
	requestDistribution = RequestDistribution{
		Op: []ChoiceWeight[correctness.OpType]{
			{Choice: correctness.OpCreate, Weight: 15},
			{Choice: correctness.OpDelete, Weight: 20},
			{Choice: correctness.OpGet, Weight: 10},
			{Choice: correctness.OpList, Weight: 25},
			{Choice: correctness.OpUpdate, Weight: 30},
		},
		Delete: DeleteDistribution{
			Preconditions: PreconditionsDistribution{
				UID: []ChoiceWeight[bool]{
					{Choice: false, Weight: 75},
					{Choice: true, Weight: 25},
				},
				ResourceVersion: []ChoiceWeight[bool]{
					{Choice: false, Weight: 75},
					{Choice: true, Weight: 25},
				},
			},
			CachedObject: []ChoiceWeight[bool]{
				{Choice: false, Weight: 75},
				{Choice: true, Weight: 25},
			},
			ValidateDeletion: []ChoiceWeight[bool]{
				{Choice: false, Weight: 75},
				{Choice: true, Weight: 25},
			},
		},
		Get: GetDistribution{
			IgnoreNotFound: []ChoiceWeight[bool]{
				{Choice: false, Weight: 50},
				{Choice: true, Weight: 50},
			},
			ResourceVersion: []ChoiceWeight[RVType]{
				{Choice: RVEmpty, Weight: 30},
				{Choice: RVZero, Weight: 15},
				{Choice: RVOne, Weight: 10},
				{Choice: RVCached, Weight: 15},
				{Choice: RVCurrent, Weight: 10},
				{Choice: RVPast, Weight: 10},
				{Choice: RVFuture, Weight: 10},
			},
		},
		List: ListDistribution{
			Scope: []ChoiceWeight[KeyScope]{
				{Choice: ScopeCluster, Weight: 50},
				{Choice: ScopeNamespace, Weight: 25},
				{Choice: ScopeObject, Weight: 25},
			},
			FieldSelector: []ChoiceWeight[FieldSelector]{
				{Choice: FieldEverything, Weight: 25},
				{Choice: FieldByName, Weight: 15},
				{Choice: FieldByNamespace, Weight: 15},
				{Choice: FieldByNode, Weight: 20},
				{Choice: FieldByEmptyNode, Weight: 15},
				{Choice: FieldCombined, Weight: 10},
			},
			LabelSelector: []ChoiceWeight[LabelSelector]{
				{Choice: LabelEverything, Weight: 60},
				{Choice: LabelByApp, Weight: 40},
			},
			ResourceVersion: []ChoiceWeight[RVType]{
				{Choice: RVEmpty, Weight: 50},
				{Choice: RVZero, Weight: 20},
				{Choice: RVCached, Weight: 30},
			},
			ResourceVersionMatch: []ChoiceWeight[metav1.ResourceVersionMatch]{
				{Choice: "", Weight: 50},
				{Choice: metav1.ResourceVersionMatchNotOlderThan, Weight: 25},
				{Choice: metav1.ResourceVersionMatchExact, Weight: 25},
			},
		},
		Update: UpdateDistribution{
			Preconditions: PreconditionsDistribution{
				UID: []ChoiceWeight[bool]{
					{Choice: false, Weight: 80},
					{Choice: true, Weight: 20},
				},
				ResourceVersion: []ChoiceWeight[bool]{
					{Choice: false, Weight: 80},
					{Choice: true, Weight: 20},
				},
			},
			NoOp: []ChoiceWeight[bool]{
				{Choice: false, Weight: 85},
				{Choice: true, Weight: 15},
			},
			CachedObject: []ChoiceWeight[bool]{
				{Choice: false, Weight: 85},
				{Choice: true, Weight: 15},
			},
			IgnoreNotFound: []ChoiceWeight[bool]{
				{Choice: false, Weight: 85},
				{Choice: true, Weight: 15},
			},
		},
	}

	unaryCfg = UnaryConfig{
		Concurrency:         8,
		Namespaces:          2,
		Objects:             4,
		MaxOperations:       10000,
		RequestDistribution: requestDistribution,
	}

	watchRequestDistribution = WatchDistribution{
		Scope: []ChoiceWeight[KeyScope]{
			{Choice: ScopeCluster, Weight: 40},
			{Choice: ScopeNamespace, Weight: 30},
			{Choice: ScopeObject, Weight: 30},
		},
		FieldSelector: []ChoiceWeight[FieldSelector]{
			{Choice: FieldEverything, Weight: 25},
			{Choice: FieldByName, Weight: 15},
			{Choice: FieldByNamespace, Weight: 15},
			{Choice: FieldByNode, Weight: 20},
			{Choice: FieldByEmptyNode, Weight: 15},
			{Choice: FieldCombined, Weight: 10},
		},
		LabelSelector: []ChoiceWeight[LabelSelector]{
			{Choice: LabelEverything, Weight: 60},
			{Choice: LabelByApp, Weight: 40},
		},
		SendInitialEvents: []ChoiceWeight[bool]{
			{Choice: false, Weight: 70},
			{Choice: true, Weight: 30},
		},
		AllowWatchBookmarks: []ChoiceWeight[bool]{
			{Choice: false, Weight: 50},
			{Choice: true, Weight: 50},
		},
		ResourceVersion: []ChoiceWeight[RVType]{
			{Choice: RVEmpty, Weight: 15},
			{Choice: RVZero, Weight: 15},
			{Choice: RVOne, Weight: 10},
			{Choice: RVCurrent, Weight: 20},
			{Choice: RVPast, Weight: 20},
			{Choice: RVFuture, Weight: 20},
		},
		WatcherBehavior: []ChoiceWeight[WatcherBehavior]{
			{Choice: WatcherFast, Weight: 40},
			{Choice: WatcherSlow, Weight: 20},
			{Choice: WatcherHiccup, Weight: 25},
			{Choice: WatcherStalled, Weight: 15},
		},
	}

	watchCfg = WatchConfig{
		Concurrency:         4,
		Duration:            500 * time.Millisecond,
		SlowDelay:           2 * time.Millisecond,
		HiccupDuration:      50 * time.Millisecond,
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
	versioner := store.Versioner()

	list := &api.PodList{}
	err := store.GetList(ctx, "/pods", storage.ListOptions{Recursive: true, Predicate: storage.Everything}, list)
	require.NoError(t, err)
	initialState, err := correctness.NewModelFromStorage(storagePrefix, list, func() runtime.Object { return &api.Pod{} }, func() runtime.Object { return &api.PodList{} }, cacheKeyFunc, versioner)
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
	linearizedOps := orderByLinearization(operations, linearizations[0][0])
	replay, err := correctness.NewReplay(initialState, linearizedOps)
	require.NoError(t, err)
	for _, op := range linearizedOps {
		require.NoError(t, replay.Validate(op.Request, op.Response))
	}
	validator := correctness.NewWatchValidator(versioner, replay, cacheKeyFunc)
	for _, w := range watches {
		require.NoError(t, validator.ValidateWatch(w.Request, w.Response))
	}
	require.Positive(t, watchEvents, "expected at least one watch event across %d watches", len(watches))
}

func orderByLinearization(ops []correctness.Operation, linearization []int) []correctness.Operation {
	ordered := make([]correctness.Operation, len(linearization))
	for i, idx := range linearization {
		ordered[i] = ops[idx]
	}
	return ordered
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
