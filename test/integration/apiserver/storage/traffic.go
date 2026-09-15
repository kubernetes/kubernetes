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
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/testing/correctness"
	api "k8s.io/kubernetes/pkg/apis/core"
)

type RequestType string

const (
	RequestTypeCreate                RequestType = "Create"
	RequestTypeDelete                RequestType = "Delete"
	RequestTypeDeleteUIDPrecondition RequestType = "DeleteUIDPrecondition"
	RequestTypeGet                   RequestType = "Get"
)

type TrafficConfig struct {
	Namespaces int
	Objects    int

	Unary UnaryConfig
	Watch WatchConfig
}

type UnaryConfig struct {
	Concurrency         int
	MaxOperations       int
	RequestDistribution []ChoiceWeight[RequestType]
}

type WatchConfig struct {
	Concurrency int
	Duration    time.Duration
	MaxEvents   int
}

type rvTracker struct {
	latest atomic.Uint64
}

func (t *rvTracker) Record(rv uint64) {
	for {
		curr := t.latest.Load()
		if rv <= curr {
			return
		}
		if t.latest.CompareAndSwap(curr, rv) {
			return
		}
	}
}

func (t *rvTracker) Latest() uint64 {
	return t.latest.Load()
}

func generateKeys(cfg TrafficConfig) []types.NamespacedName {
	numNamespaces := cfg.Namespaces
	if numNamespaces <= 0 {
		numNamespaces = 1
	}
	keys := make([]types.NamespacedName, 0, cfg.Objects)
	for i := 0; i < cfg.Objects; i++ {
		ns := fmt.Sprintf("ns-%d", (i%numNamespaces)+1)
		name := fmt.Sprintf("pod-%d", (i/numNamespaces)+1)
		keys = append(keys, types.NamespacedName{
			Namespace: ns,
			Name:      name,
		})
	}
	return keys
}

// RunTraffic drives concurrent storage operations and records all invocations.
func RunTraffic(ctx context.Context, store storage.Interface, cfg TrafficConfig) ([]correctness.Operation, []correctness.RecordedWatch, error) {
	if cfg.Unary.Concurrency <= 0 {
		return nil, nil, fmt.Errorf("unary concurrency must be positive")
	}
	if cfg.Objects <= 0 {
		return nil, nil, fmt.Errorf("objects must be positive")
	}
	if cfg.Namespaces <= 0 {
		return nil, nil, fmt.Errorf("namespaces must be positive")
	}
	if len(cfg.Unary.RequestDistribution) == 0 {
		return nil, nil, fmt.Errorf("operations must be non-empty")
	}
	if cfg.Unary.MaxOperations <= 0 {
		return nil, nil, fmt.Errorf("MaxOperations must be positive")
	}

	keys := generateKeys(cfg)
	var requestCounter atomic.Int64
	var unaryMu sync.Mutex
	var operations []correctness.Operation

	trafficCtx, cancelTraffic := context.WithCancel(ctx)
	defer cancelTraffic()

	rvTracker := &rvTracker{}

	var watchMu sync.Mutex
	var recordedWatches []correctness.RecordedWatch
	var watchWg sync.WaitGroup

	if cfg.Watch.Concurrency > 0 {
		watchDuration := cfg.Watch.Duration
		if watchDuration <= 0 {
			watchDuration = 100 * time.Millisecond
		}
		for wid := 0; wid < cfg.Watch.Concurrency; wid++ {
			watchWg.Add(1)
			go func(id int) {
				defer watchWg.Done()
				for {
					select {
					case <-trafficCtx.Done():
						return
					default:
					}
					req := randomWatchRequest(keys, rvTracker.Latest())
					recWatch, err := runWatchSession(trafficCtx, store, req, watchDuration, cfg.Watch.MaxEvents)
					if err == nil {
						watchMu.Lock()
						recordedWatches = append(recordedWatches, recWatch)
						watchMu.Unlock()
					}
				}
			}(wid)
		}
	}

	var unaryWg sync.WaitGroup
	for clientID := 0; clientID < cfg.Unary.Concurrency; clientID++ {
		unaryWg.Add(1)
		go func(cid int) {
			defer unaryWg.Done()
			var cachedObj runtime.Object
			for {
				select {
				case <-trafficCtx.Done():
					return
				default:
				}
				request := randomRequest(keys, cfg.Unary.RequestDistribution, cachedObj)
				if request == nil {
					continue
				}
				requestNumber := requestCounter.Add(1)
				if cfg.Unary.MaxOperations > 0 && requestNumber > int64(cfg.Unary.MaxOperations) {
					return
				}
				start := time.Now()
				response := runTraffic(ctx, store, request)
				end := time.Now()
				if response.Object != nil {
					cachedObj = response.Object
					if acc, err := meta.Accessor(response.Object); err == nil {
						if rv, err := store.Versioner().ParseResourceVersion(acc.GetResourceVersion()); err == nil {
							rvTracker.Record(rv)
						}
					}
				}

				op := correctness.Operation{
					ClientID: cid,
					Start:    start,
					End:      end,
					Request:  *request,
					Response: response,
				}

				unaryMu.Lock()
				operations = append(operations, op)
				unaryMu.Unlock()
			}
		}(clientID)
	}

	unaryWg.Wait()
	cancelTraffic()
	watchWg.Wait()

	return operations, recordedWatches, nil
}

func runWatchSession(ctx context.Context, store storage.Interface, req correctness.WatchRequest, duration time.Duration, maxEvents int) (correctness.RecordedWatch, error) {
	rec, err := correctness.NewWatchRecorder(ctx, store, req)
	if err != nil {
		return correctness.RecordedWatch{}, err
	}

	timer := time.NewTimer(duration)
	defer timer.Stop()

	ticker := time.NewTicker(10 * time.Millisecond)
	defer ticker.Stop()

	done := false
	for !done {
		select {
		case <-ctx.Done():
			done = true
		case <-timer.C:
			done = true
		case <-ticker.C:
			if maxEvents > 0 && len(rec.Events()) >= maxEvents {
				done = true
			}
		}
	}

	rec.Stop()
	return rec.RecordedWatch(), nil
}

func randomRequest(keys []types.NamespacedName, ops []ChoiceWeight[RequestType], cached runtime.Object) *correctness.Request {
	selectedOp := PickRandom(ops)
	key := keys[rand.Intn(len(keys))]

	switch selectedOp {
	case RequestTypeCreate:
		obj := validPod(key.Namespace, key.Name)
		return &correctness.Request{
			Op:  correctness.OpCreate,
			Key: storageKey(key),
			Create: correctness.CreateRequest{
				Object: obj,
			},
		}
	case RequestTypeDelete:
		return &correctness.Request{
			Op:  correctness.OpDelete,
			Key: storageKey(key),
		}
	case RequestTypeDeleteUIDPrecondition:
		if cached == nil {
			return nil
		}
		accessor, err := meta.Accessor(cached)
		if err != nil {
			panic(err)
		}
		uid := accessor.GetUID()
		return &correctness.Request{
			Op:  correctness.OpDelete,
			Key: storageKey(key),
			Delete: correctness.DeleteRequest{
				Preconditions: &storage.Preconditions{UID: &uid},
			},
		}
	case RequestTypeGet:
		getOpts := storage.GetOptions{}
		return &correctness.Request{
			Op:  correctness.OpGet,
			Key: storageKey(key),
			Get: correctness.GetRequest{
				Options: getOpts,
			},
		}
	default:
		panic(fmt.Sprintf("%v: unknown operation", selectedOp))
	}
}

func storageKey(key types.NamespacedName) string {
	return "/pods/" + key.String()
}

func runTraffic(ctx context.Context, store storage.Interface, request *correctness.Request) correctness.Response {
	out := &api.Pod{}
	var err error
	key := request.Key
	switch request.Op {
	case correctness.OpCreate:
		err = store.Create(ctx, key, request.Create.Object, out, 0)
	case correctness.OpDelete:
		err = store.Delete(ctx, key, out, request.Delete.Preconditions, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{})
	case correctness.OpGet:
		err = store.Get(ctx, key, request.Get.Options, out)
	default:
		panic(fmt.Sprintf("%v: unknown operation", request.Op))
	}
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return correctness.Response{
				Err: err,
			}
		}
		if _, ok := errors.AsType[*storage.StorageError](err); ok {
			return correctness.Response{
				Err: err,
			}
		}
		panic(err)
	}
	response := correctness.Response{
		Object: out,
	}
	return response
}

func validPod(namespace, name string) *api.Pod {
	gracePeriod := int64(30)
	enableServiceLinks := true
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
			UID:       uuid.NewUUID(),
		},
		Spec: api.PodSpec{
			RestartPolicy:                 api.RestartPolicyAlways,
			TerminationGracePeriodSeconds: &gracePeriod,
			DNSPolicy:                     api.DNSClusterFirst,
			SecurityContext:               &api.PodSecurityContext{},
			SchedulerName:                 "default-scheduler",
			EnableServiceLinks:            &enableServiceLinks,
		},
	}
}
