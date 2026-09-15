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
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/testing/correctness"
	api "k8s.io/kubernetes/pkg/apis/core"
)

type RequestType string

const (
	RequestTypeCreate                 RequestType = "Create"
	RequestTypeDelete                 RequestType = "Delete"
	RequestTypeDeleteUIDPrecondition  RequestType = "DeleteUIDPrecondition"
	RequestTypeGet                    RequestType = "Get"
	RequestTypeUpdate                 RequestType = "Update"
	RequestTypeUpdateUIDPrecondition  RequestType = "UpdateUIDPrecondition"
	RequestTypeUpdateNoOp             RequestType = "UpdateNoOp"
	RequestTypeUpdateWithCachedObject RequestType = "UpdateWithCachedObject"
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
func RunTraffic(ctx context.Context, store storage.Interface, cfg TrafficConfig) ([]correctness.Operation, []correctness.WatchOperation, error) {
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

	// Closed once the unary workload is done. Watch workers stop opening new
	// sessions but let the in-flight one run to its own deadline, so the events
	// it already collected are validated instead of discarded.
	stopWatches := make(chan struct{})
	var stopWatchesOnce sync.Once
	defer stopWatchesOnce.Do(func() { close(stopWatches) })

	rvTracker := &rvTracker{}

	var watchMu sync.Mutex
	var recordedWatches []correctness.WatchOperation
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
					case <-ctx.Done():
						return
					case <-stopWatches:
						return
					default:
					}
					req := randomWatchRequest(rvTracker.Latest())
					recWatch, err := runWatchSession(ctx, store, req, watchDuration, cfg.Watch.MaxEvents)
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
				case <-ctx.Done():
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
	// Bounded by cfg.Watch.Duration: at most one more session per watch worker.
	stopWatchesOnce.Do(func() { close(stopWatches) })
	watchWg.Wait()

	return operations, recordedWatches, nil
}

// watchPrefix is the only scope the validator can express: it replays the whole
// operation history, so a narrower watch would expect events it never receives.
const watchPrefix = "/pods/"

func runWatchSession(ctx context.Context, store storage.Interface, req correctness.WatchRequest, duration time.Duration, maxEvents int) (correctness.WatchOperation, error) {
	// Without this the cacher answers ResourceVersion="0" with a synthetic dump
	// of its store, which is not part of the operation history.
	sendInitialEvents := false
	w, err := store.Watch(ctx, watchPrefix, storage.ListOptions{
		ResourceVersion:   req.ResourceVersion,
		Predicate:         storage.Everything,
		Recursive:         true,
		SendInitialEvents: &sendInitialEvents,
	})
	if err != nil {
		return correctness.WatchOperation{}, err
	}
	defer w.Stop()

	timer := time.NewTimer(duration)
	defer timer.Stop()

	var events []watch.Event
	for {
		select {
		case <-ctx.Done():
			return correctness.WatchOperation{Request: req, Response: correctness.WatchResponse{Events: events}}, nil
		case <-timer.C:
			return correctness.WatchOperation{Request: req, Response: correctness.WatchResponse{Events: events}}, nil
		case event, open := <-w.ResultChan():
			if !open {
				return correctness.WatchOperation{Request: req, Response: correctness.WatchResponse{Events: events}}, nil
			}
			events = append(events, correctness.UnwrapEvent(event))
			if maxEvents > 0 && len(events) >= maxEvents {
				return correctness.WatchOperation{Request: req, Response: correctness.WatchResponse{Events: events}}, nil
			}
		}
	}
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
	case RequestTypeUpdate:
		version := fmt.Sprintf("%d", rand.Intn(10000))
		return &correctness.Request{
			Op:  correctness.OpUpdate,
			Key: storageKey(key),
			Update: correctness.UpdateRequest{
				IgnoreNotFound: false,
				UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
					pod := obj.(*api.Pod).DeepCopy()
					if pod.Annotations == nil {
						pod.Annotations = make(map[string]string)
					}
					pod.Annotations["version"] = version
					return pod, nil
				}),
			},
		}
	case RequestTypeUpdateUIDPrecondition:
		if cached == nil {
			return nil
		}
		accessor, err := meta.Accessor(cached)
		if err != nil {
			panic(err)
		}
		uid := accessor.GetUID()
		version := fmt.Sprintf("%d", rand.Intn(10000))
		return &correctness.Request{
			Op:  correctness.OpUpdate,
			Key: storageKey(key),
			Update: correctness.UpdateRequest{
				IgnoreNotFound: false,
				Preconditions:  &storage.Preconditions{UID: &uid},
				UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
					pod := obj.(*api.Pod).DeepCopy()
					if pod.Annotations == nil {
						pod.Annotations = make(map[string]string)
					}
					pod.Annotations["version"] = version
					return pod, nil
				}),
			},
		}
	case RequestTypeUpdateNoOp:
		return &correctness.Request{
			Op:  correctness.OpUpdate,
			Key: storageKey(key),
			Update: correctness.UpdateRequest{
				IgnoreNotFound: false,
				UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
					return obj.(*api.Pod).DeepCopy(), nil
				}),
			},
		}
	case RequestTypeUpdateWithCachedObject:
		if cached == nil {
			return nil
		}
		version := fmt.Sprintf("%d", rand.Intn(10000))
		return &correctness.Request{
			Op:  correctness.OpUpdate,
			Key: storageKey(key),
			Update: correctness.UpdateRequest{
				IgnoreNotFound:       false,
				CachedExistingObject: cached.DeepCopyObject(),
				UpdateFunc: storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
					pod := obj.(*api.Pod).DeepCopy()
					if pod.Annotations == nil {
						pod.Annotations = make(map[string]string)
					}
					pod.Annotations["version"] = version
					return pod, nil
				}),
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
	case correctness.OpUpdate:
		err = store.GuaranteedUpdate(ctx, key, out, request.Update.IgnoreNotFound, request.Update.Preconditions, request.Update.UpdateFunc, request.Update.CachedExistingObject)
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
