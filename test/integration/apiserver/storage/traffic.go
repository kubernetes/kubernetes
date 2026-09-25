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
	"strconv"
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

// WatchRequestType selects the resource version a watch starts from.
type WatchRequestType string

const (
	RVEmpty   WatchRequestType = "RVEmpty"
	RVZero    WatchRequestType = "RVZero"
	RVOne     WatchRequestType = "RVOne"
	RVCurrent WatchRequestType = "RVCurrent"
	RVPast    WatchRequestType = "RVPast"
	RVFuture  WatchRequestType = "RVFuture"
)

type UnaryConfig struct {
	Concurrency         int
	MaxOperations       int
	Namespaces          int
	Objects             int
	RequestDistribution []ChoiceWeight[RequestType]
}

type WatchConfig struct {
	Concurrency         int
	Duration            time.Duration
	MaxEvents           int
	RequestDistribution []ChoiceWeight[WatchRequestType]
}

func generateKeys(cfg UnaryConfig) []types.NamespacedName {
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

// RunUnaryTraffic drives concurrent storage operations and records all invocations.
func RunUnaryTraffic(ctx context.Context, store storage.Interface, cfg UnaryConfig) ([]correctness.Operation, error) {
	if cfg.Concurrency <= 0 {
		return nil, fmt.Errorf("concurrency must be positive")
	}
	if cfg.Objects <= 0 {
		return nil, fmt.Errorf("objects must be positive")
	}
	if cfg.Namespaces <= 0 {
		return nil, fmt.Errorf("namespaces must be positive")
	}
	if len(cfg.RequestDistribution) == 0 {
		return nil, fmt.Errorf("operations must be non-empty")
	}
	if cfg.MaxOperations <= 0 {
		return nil, fmt.Errorf("MaxOperations must be positive")
	}

	keys := generateKeys(cfg)
	var requestCounter atomic.Int64
	var mu sync.Mutex
	var operations []correctness.Operation

	var wg sync.WaitGroup
	for clientID := 0; clientID < cfg.Concurrency; clientID++ {
		wg.Add(1)
		go func(cid int) {
			defer wg.Done()
			var cachedObj runtime.Object
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}
				request := randomRequest(keys, cfg.RequestDistribution, cachedObj)
				if request == nil {
					continue
				}
				requestNumber := requestCounter.Add(1)
				if cfg.MaxOperations > 0 && requestNumber > int64(cfg.MaxOperations) {
					return
				}
				start := time.Now()
				response := runTraffic(ctx, store, request)
				end := time.Now()
				if response.Object != nil {
					cachedObj = response.Object
				}

				op := correctness.Operation{
					ClientID: cid,
					Start:    start,
					End:      end,
					Request:  *request,
					Response: response,
				}

				mu.Lock()
				operations = append(operations, op)
				mu.Unlock()
			}
		}(clientID)
	}

	wg.Wait()
	return operations, nil
}

// RunWatchTraffic keeps cfg.Concurrency watches open until stop is closed and
// records what each of them received. Watches still open at that point run to
// their own deadline, so the events they already collected are not discarded.
func RunWatchTraffic(ctx context.Context, store storage.Interface, cfg WatchConfig, stop <-chan struct{}) []correctness.WatchOperation {
	var mu sync.Mutex
	var watches []correctness.WatchOperation
	var wg sync.WaitGroup
	for range cfg.Concurrency {
		wg.Go(func() {
			for {
				select {
				case <-ctx.Done():
					return
				case <-stop:
					return
				default:
				}
				request := randomWatchRequest(ctx, store, cfg.RequestDistribution)
				response := runWatch(ctx, store, request, cfg)

				mu.Lock()
				watches = append(watches, correctness.WatchOperation{Request: request, Response: response})
				mu.Unlock()
			}
		})
	}
	wg.Wait()
	return watches
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

func randomWatchRequest(ctx context.Context, store storage.Interface, distribution []ChoiceWeight[WatchRequestType]) correctness.WatchRequest {
	var offset int64
	switch selected := PickRandom(distribution); selected {
	case RVEmpty:
		return correctness.WatchRequest{ResourceVersion: ""}
	case RVZero:
		return correctness.WatchRequest{ResourceVersion: "0"}
	case RVOne:
		return correctness.WatchRequest{ResourceVersion: "1"}
	case RVCurrent:
		offset = 0
	case RVPast:
		offset = -int64(1 + rand.Intn(10))
	case RVFuture:
		offset = int64(1 + rand.Intn(10))
	default:
		panic(fmt.Sprintf("%v: unknown watch request type", selected))
	}
	currentRV, err := store.GetCurrentResourceVersion(ctx)
	if err != nil {
		panic(err)
	}
	rv := max(int64(currentRV)+offset, 1)
	return correctness.WatchRequest{ResourceVersion: strconv.FormatInt(rv, 10)}
}

func runWatch(ctx context.Context, store storage.Interface, req correctness.WatchRequest, cfg WatchConfig) correctness.WatchResponse {
	// Without SendInitialEvents=false, ResourceVersion="0" or "" causes storage
	// to emit synthetic ADDED events for existing objects at their current RV.
	sendInitialEvents := false
	w, err := store.Watch(ctx, "/pods/", storage.ListOptions{
		ResourceVersion:   req.ResourceVersion,
		Predicate:         storage.Everything,
		Recursive:         true,
		SendInitialEvents: &sendInitialEvents,
	})
	if err != nil {
		if _, ok := errors.AsType[*storage.StorageError](err); ok {
			return correctness.WatchResponse{Err: err}
		}
		panic(err)
	}
	defer w.Stop()

	timer := time.NewTimer(cfg.Duration)
	defer timer.Stop()

	var events []watch.Event
	for {
		select {
		case <-ctx.Done():
			return correctness.WatchResponse{Events: events, Err: ctx.Err()}
		case <-timer.C:
			return correctness.WatchResponse{Events: events}
		case event, open := <-w.ResultChan():
			if !open {
				return correctness.WatchResponse{Events: events}
			}
			if cacheable, ok := event.Object.(runtime.CacheableObject); ok {
				event.Object = cacheable.GetObject()
			}
			events = append(events, event)
			if cfg.MaxEvents > 0 && len(events) >= cfg.MaxEvents {
				return correctness.WatchResponse{Events: events}
			}
		}
	}
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
