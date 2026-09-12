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

type TraffiConfig struct {
	Concurrency         int
	MaxOperations       int
	Namespaces          int
	Objects             int
	RequestDistribution []ChoiceWeight[RequestType]
}

func generateKeys(cfg TraffiConfig) []types.NamespacedName {
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
func RunTraffic(ctx context.Context, store storage.Interface, cfg TraffiConfig) ([]correctness.Operation, error) {
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

func randomRequest(keys []types.NamespacedName, ops []ChoiceWeight[RequestType], cached runtime.Object) *correctness.Request {
	selectedOp := PickRandom(ops)
	key := keys[rand.Intn(len(keys))]

	switch selectedOp {
	case RequestTypeCreate:
		obj := validPod(key.Namespace, key.Name)
		return &correctness.Request{
			Op:     correctness.OpCreate,
			Key:    storageKey(key),
			Object: obj,
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
			Op:            correctness.OpDelete,
			Key:           storageKey(key),
			Preconditions: &storage.Preconditions{UID: &uid},
		}
	case RequestTypeGet:
		getOpts := storage.GetOptions{}
		return &correctness.Request{
			Op:         correctness.OpGet,
			Key:        storageKey(key),
			GetOptions: getOpts,
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
		err = store.Create(ctx, key, request.Object, out, 0)
	case correctness.OpDelete:
		err = store.Delete(ctx, key, out, request.Preconditions, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{})
	case correctness.OpGet:
		err = store.Get(ctx, key, request.GetOptions, out)
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
