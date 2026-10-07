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
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/testing/correctness"
	api "k8s.io/kubernetes/pkg/apis/core"
	registrypod "k8s.io/kubernetes/pkg/registry/core/pod"
)

type KeyScope string

const (
	ScopeCluster   KeyScope = "Cluster"
	ScopeNamespace KeyScope = "Namespace"
	ScopeObject    KeyScope = "Object"
)

type RVType string

const (
	RVEmpty   RVType = "Empty"
	RVZero    RVType = "Zero"
	RVOne     RVType = "One"
	RVCached  RVType = "Cached"
	RVCurrent RVType = "Current"
	RVPast    RVType = "Past"
	RVFuture  RVType = "Future"
)

type FieldSelector string

const (
	FieldEverything  FieldSelector = "Everything"
	FieldByName      FieldSelector = "ByName"
	FieldByNamespace FieldSelector = "ByNamespace"
	FieldByNode      FieldSelector = "ByNode"
	FieldByEmptyNode FieldSelector = "ByEmptyNode"
	FieldCombined    FieldSelector = "Combined"
)

type LabelSelector string

const (
	LabelEverything LabelSelector = "Everything"
	LabelByApp      LabelSelector = "ByApp"
)

type WatcherBehavior string

const (
	WatcherFast    WatcherBehavior = "Fast"
	WatcherSlow    WatcherBehavior = "Slow"
	WatcherHiccup  WatcherBehavior = "Hiccup"
	WatcherStalled WatcherBehavior = "Stalled"
)

var (
	watchNamespaces = []string{"ns-1", "ns-2"}
	watchPodNames   = []string{"pod-1", "pod-2"}
	nodeNames       = []string{"", "node-1", "node-2"}
	nonEmptyNodes   = []string{"node-1", "node-2"}
	appLabels       = []string{"", "app-a", "app-b"}
	nonEmptyApps    = []string{"app-a", "app-b"}
)

type RequestDistribution struct {
	Op     []ChoiceWeight[correctness.OpType]
	Get    GetDistribution
	List   ListDistribution
	Update UpdateDistribution
	Delete DeleteDistribution
}

type GetDistribution struct {
	IgnoreNotFound  []ChoiceWeight[bool]
	ResourceVersion []ChoiceWeight[RVType]
}

type ListDistribution struct {
	Scope                []ChoiceWeight[KeyScope]
	FieldSelector        []ChoiceWeight[FieldSelector]
	LabelSelector        []ChoiceWeight[LabelSelector]
	ResourceVersion      []ChoiceWeight[RVType]
	ResourceVersionMatch []ChoiceWeight[metav1.ResourceVersionMatch]
}

type UpdateDistribution struct {
	Preconditions  PreconditionsDistribution
	NoOp           []ChoiceWeight[bool]
	CachedObject   []ChoiceWeight[bool]
	IgnoreNotFound []ChoiceWeight[bool]
}

type DeleteDistribution struct {
	Preconditions PreconditionsDistribution
}

type PreconditionsDistribution struct {
	UID             []ChoiceWeight[bool]
	ResourceVersion []ChoiceWeight[bool]
}

type WatchDistribution struct {
	Scope               []ChoiceWeight[KeyScope]
	FieldSelector       []ChoiceWeight[FieldSelector]
	LabelSelector       []ChoiceWeight[LabelSelector]
	SendInitialEvents   []ChoiceWeight[bool]
	AllowWatchBookmarks []ChoiceWeight[bool]
	ResourceVersion     []ChoiceWeight[RVType]
	WatcherBehavior     []ChoiceWeight[WatcherBehavior]
}

type UnaryConfig struct {
	Concurrency         int
	MaxOperations       int
	Namespaces          int
	Objects             int
	RequestDistribution RequestDistribution
}

type WatchConfig struct {
	Concurrency         int
	Duration            time.Duration
	SlowDelay           time.Duration
	HiccupDuration      time.Duration
	MaxEvents           int
	RequestDistribution WatchDistribution
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
	if len(cfg.RequestDistribution.Op) == 0 {
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
				request := randomRequest(ctx, store, keys, cfg.RequestDistribution, cachedObj)
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
				if response.Object != nil && request.Op != correctness.OpList && response.Object.(*api.Pod).Name != "" {
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
				mode := WatcherFast
				if len(cfg.RequestDistribution.WatcherBehavior) > 0 {
					mode = PickRandom(cfg.RequestDistribution.WatcherBehavior)
				}
				response := runWatch(ctx, store, request, mode, cfg)

				mu.Lock()
				watches = append(watches, correctness.WatchOperation{Request: request, Response: response})
				mu.Unlock()
			}
		})
	}
	wg.Wait()
	return watches
}

func randomRequest(ctx context.Context, store storage.Interface, keys []types.NamespacedName, dist RequestDistribution, cached runtime.Object) *correctness.Request {
	key := keys[rand.Intn(len(keys))]

	switch selectedOp := PickRandom(dist.Op); selectedOp {
	case correctness.OpCreate:
		obj := validPod(key.Namespace, key.Name)
		obj.UID = uuid.NewUUID()
		return &correctness.Request{
			Op:  correctness.OpCreate,
			Key: storageKey(key),
			Create: correctness.CreateRequest{
				Object: obj,
			},
		}
	case correctness.OpDelete:
		preconditions, ok := pickPreconditions(dist.Delete.Preconditions, cached)
		if !ok {
			return nil
		}
		return &correctness.Request{
			Op:  correctness.OpDelete,
			Key: storageKey(key),
			Delete: correctness.DeleteRequest{
				Preconditions: preconditions,
			},
		}
	case correctness.OpGet:
		rv, ok := pickResourceVersion(ctx, store, dist.Get.ResourceVersion, cached)
		if !ok {
			return nil
		}
		return &correctness.Request{
			Op:  correctness.OpGet,
			Key: storageKey(key),
			Get: correctness.GetRequest{
				Options: storage.GetOptions{
					IgnoreNotFound:  PickRandom(dist.Get.IgnoreNotFound),
					ResourceVersion: rv,
				},
			},
		}
	case correctness.OpList:
		opts := storage.ListOptions{
			Predicate: pickPredicate(dist.List.FieldSelector, dist.List.LabelSelector),
		}
		var listKey string
		switch scope := PickRandom(dist.List.Scope); scope {
		case ScopeCluster:
			listKey, opts.Recursive = "/pods/", true
		case ScopeNamespace:
			listKey, opts.Recursive = "/pods/"+key.Namespace, true
		case ScopeObject:
			listKey, opts.Recursive = storageKey(key), false
		default:
			panic(fmt.Sprintf("%v: unknown list scope", scope))
		}
		rv, ok := pickResourceVersion(ctx, store, dist.List.ResourceVersion, cached)
		if !ok {
			return nil
		}
		opts.ResourceVersion = rv
		if rv != "" {
			opts.ResourceVersionMatch = PickRandom(dist.List.ResourceVersionMatch)
			if rv == "0" && opts.ResourceVersionMatch == metav1.ResourceVersionMatchExact {
				return nil
			}
		}
		return &correctness.Request{
			Op:   correctness.OpList,
			Key:  listKey,
			List: correctness.ListRequest{Options: opts},
		}
	case correctness.OpUpdate:
		preconditions, ok := pickPreconditions(dist.Update.Preconditions, cached)
		if !ok {
			return nil
		}
		useCached := PickRandom(dist.Update.CachedObject)
		if useCached && cached == nil {
			return nil
		}
		ignoreNotFound := PickRandom(dist.Update.IgnoreNotFound)
		noOp := PickRandom(dist.Update.NoOp)
		if ignoreNotFound && noOp {
			return nil
		}
		var cachedExisting runtime.Object
		if useCached {
			cachedExisting = cached.DeepCopyObject()
		}
		updateFn := randomUpdate(key)
		if noOp {
			updateFn = storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
				return obj.(*api.Pod).DeepCopy(), nil
			})
		}
		return &correctness.Request{
			Op:  correctness.OpUpdate,
			Key: storageKey(key),
			Update: correctness.UpdateRequest{
				IgnoreNotFound:       ignoreNotFound,
				Preconditions:        preconditions,
				CachedExistingObject: cachedExisting,
				UpdateFunc:           updateFn,
			},
		}
	default:
		panic(fmt.Sprintf("%v: unknown operation", selectedOp))
	}
}

func pickResourceVersion(ctx context.Context, store storage.Interface, dist []ChoiceWeight[RVType], cached runtime.Object) (string, bool) {
	switch rvType := PickRandom(dist); rvType {
	case RVEmpty:
		return "", true
	case RVZero:
		return "0", true
	case RVOne:
		return "1", true
	case RVCached:
		if cached == nil {
			return "", false
		}
		accessor, err := meta.Accessor(cached)
		if err != nil {
			panic(err)
		}
		return accessor.GetResourceVersion(), true
	case RVCurrent:
		return relativeRV(ctx, store, 0), true
	case RVPast:
		return relativeRV(ctx, store, -int64(1+rand.Intn(10))), true
	case RVFuture:
		return relativeRV(ctx, store, int64(1+rand.Intn(10))), true
	default:
		panic(fmt.Sprintf("%v: unknown RVType", rvType))
	}
}

func pickPreconditions(dist PreconditionsDistribution, cached runtime.Object) (*storage.Preconditions, bool) {
	useUID := PickRandom(dist.UID)
	useRV := PickRandom(dist.ResourceVersion)
	if !useUID && !useRV {
		return nil, true
	}
	if cached == nil {
		return nil, false
	}
	accessor, err := meta.Accessor(cached)
	if err != nil {
		panic(err)
	}
	var p storage.Preconditions
	if useUID {
		uid := accessor.GetUID()
		p.UID = &uid
	}
	if useRV {
		rv := accessor.GetResourceVersion()
		p.ResourceVersion = &rv
	}
	return &p, true
}

func storageKey(key types.NamespacedName) string {
	return "/pods/" + key.String()
}

func randomUpdate(key types.NamespacedName) storage.UpdateFunc {
	version := strconv.Itoa(rand.Intn(10000))
	nodeName := nodeNames[rand.Intn(len(nodeNames))]
	appLabel := appLabels[rand.Intn(len(appLabels))]
	return func(obj runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		pod := obj.(*api.Pod).DeepCopy()
		if pod.Name == "" {
			pod = validPod(key.Namespace, key.Name)
		}
		if pod.Annotations == nil {
			pod.Annotations = make(map[string]string)
		}
		pod.Annotations["version"] = version
		pod.Spec.NodeName = nodeName
		if appLabel == "" {
			pod.Labels = nil
		} else {
			pod.Labels = map[string]string{"app": appLabel}
		}
		return pod, nil, nil
	}
}

func runTraffic(ctx context.Context, store storage.Interface, request *correctness.Request) correctness.Response {
	var out runtime.Object = &api.Pod{}
	var err error
	key := request.Key
	switch request.Op {
	case correctness.OpCreate:
		err = store.Create(ctx, key, request.Create.Object, out, 0)
	case correctness.OpDelete:
		err = store.Delete(ctx, key, out, request.Delete.Preconditions, storage.ValidateAllObjectFunc, nil, storage.DeleteOptions{})
	case correctness.OpGet:
		err = store.Get(ctx, key, request.Get.Options, out)
	case correctness.OpList:
		out = &api.PodList{}
		err = store.GetList(ctx, key, request.List.Options, out)
	case correctness.OpUpdate:
		err = store.GuaranteedUpdate(ctx, key, out, request.Update.IgnoreNotFound, request.Update.Preconditions, request.Update.UpdateFunc, request.Update.CachedExistingObject)
	default:
		panic(fmt.Sprintf("%v: unknown operation", request.Op))
	}
	if err != nil {
		if _, ok := errors.AsType[*storage.StorageError](err); ok || storage.IsTooLargeResourceVersion(err) {
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

func randomWatchRequest(ctx context.Context, store storage.Interface, distribution WatchDistribution) correctness.WatchRequest {
	var rv string
	watchList := PickRandom(distribution.SendInitialEvents)
	selected := PickRandom(distribution.ResourceVersion)
	for watchList && selected == RVFuture {
		selected = PickRandom(distribution.ResourceVersion)
	}
	switch selected {
	case RVEmpty:
		rv = ""
	case RVZero:
		rv = "0"
	case RVOne:
		rv = "1"
	case RVCurrent:
		rv = relativeRV(ctx, store, 0)
	case RVPast:
		rv = relativeRV(ctx, store, -int64(1+rand.Intn(10)))
	case RVFuture:
		rv = relativeRV(ctx, store, int64(1+rand.Intn(10)))
	default:
		panic(fmt.Sprintf("%v: unknown watch request type", selected))
	}

	ns := watchNamespaces[rand.Intn(len(watchNamespaces))]
	name := watchPodNames[rand.Intn(len(watchPodNames))]

	watchKey := "/pods/"
	recursive := true
	if len(distribution.Scope) > 0 {
		switch scope := PickRandom(distribution.Scope); scope {
		case ScopeCluster:
			watchKey, recursive = "/pods/", true
		case ScopeNamespace:
			watchKey, recursive = "/pods/"+ns, true
		case ScopeObject:
			watchKey, recursive = "/pods/"+ns+"/"+name, false
		default:
			panic(fmt.Sprintf("%v: unknown watch scope", scope))
		}
	}

	pred := pickPredicate(distribution.FieldSelector, distribution.LabelSelector)
	if len(distribution.AllowWatchBookmarks) > 0 {
		pred.AllowWatchBookmarks = PickRandom(distribution.AllowWatchBookmarks)
	}

	opts := storage.ListOptions{ResourceVersion: rv, Predicate: pred, Recursive: recursive}
	switch {
	case watchList:
		opts.Predicate.AllowWatchBookmarks = true
		opts.SendInitialEvents = new(true)
		opts.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan
	case rv == "" || rv == "0":
		// Otherwise storage starts with synthetic ADDED events for existing
		// objects. API validation requires the match with sendInitialEvents.
		opts.SendInitialEvents = new(false)
		opts.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan
	}
	return correctness.WatchRequest{Key: watchKey, Options: opts}
}

func pickPredicate(fieldDist []ChoiceWeight[FieldSelector], labelDist []ChoiceWeight[LabelSelector]) storage.SelectionPredicate {
	ns := watchNamespaces[rand.Intn(len(watchNamespaces))]
	name := watchPodNames[rand.Intn(len(watchPodNames))]
	node := nonEmptyNodes[rand.Intn(len(nonEmptyNodes))]
	app := nonEmptyApps[rand.Intn(len(nonEmptyApps))]

	fieldSel := fields.Everything()
	var indexFields []string
	if len(fieldDist) > 0 {
		switch fs := PickRandom(fieldDist); fs {
		case FieldEverything:
			fieldSel = fields.Everything()
		case FieldByName:
			fieldSel = fields.OneTermEqualSelector("metadata.name", name)
		case FieldByNamespace:
			fieldSel = fields.OneTermEqualSelector("metadata.namespace", ns)
		case FieldByNode:
			fieldSel = fields.OneTermEqualSelector("spec.nodeName", node)
			indexFields = []string{"spec.nodeName"}
		case FieldByEmptyNode:
			fieldSel = fields.OneTermEqualSelector("spec.nodeName", "")
			indexFields = []string{"spec.nodeName"}
		case FieldCombined:
			fieldSel = fields.AndSelectors(
				fields.OneTermEqualSelector("metadata.namespace", ns),
				fields.OneTermEqualSelector("spec.nodeName", node),
			)
			indexFields = []string{"spec.nodeName"}
		default:
			panic(fmt.Sprintf("%v: unknown field selector", fs))
		}
	}

	labelSel := labels.Everything()
	if len(labelDist) > 0 {
		switch ls := PickRandom(labelDist); ls {
		case LabelEverything:
			labelSel = labels.Everything()
		case LabelByApp:
			labelSel = labels.SelectorFromSet(labels.Set{"app": app})
		default:
			panic(fmt.Sprintf("%v: unknown label selector", ls))
		}
	}

	return storage.SelectionPredicate{
		Label:       labelSel,
		Field:       fieldSel,
		GetAttrs:    registrypod.GetAttrs,
		IndexFields: indexFields,
	}
}

func relativeRV(ctx context.Context, store storage.Interface, offset int64) string {
	currentRV, err := store.GetCurrentResourceVersion(ctx)
	if err != nil {
		panic(err)
	}
	return strconv.FormatInt(max(int64(currentRV)+offset, 1), 10)
}

func runWatch(ctx context.Context, store storage.Interface, req correctness.WatchRequest, mode WatcherBehavior, cfg WatchConfig) correctness.WatchResponse {
	watchCtx := ctx
	switch parts := strings.Split(strings.Trim(req.Key, "/"), "/"); len(parts) {
	case 2:
		watchCtx = genericapirequest.WithNamespace(watchCtx, parts[1])
	case 3:
		watchCtx = genericapirequest.WithNamespace(watchCtx, parts[1])
		watchCtx = genericapirequest.WithRequestInfo(watchCtx, &genericapirequest.RequestInfo{Name: parts[2]})
	}

	w, err := store.Watch(watchCtx, req.Key, req.Options)
	if err != nil {
		if _, ok := errors.AsType[*storage.StorageError](err); ok {
			return correctness.WatchResponse{Err: err}
		}
		panic(err)
	}
	defer w.Stop()

	timer := time.NewTimer(cfg.Duration)
	defer timer.Stop()

	if mode == WatcherStalled {
		stalledTimer := timer.C
		if cfg.HiccupDuration > 0 {
			stalledTimer = time.After(cfg.HiccupDuration)
		}
		select {
		case <-ctx.Done():
			return correctness.WatchResponse{Err: ctx.Err()}
		case <-stalledTimer:
			return correctness.WatchResponse{}
		}
	}

	initialEventsDone := req.Options.SendInitialEvents == nil || !*req.Options.SendInitialEvents
	hiccupDone := mode != WatcherHiccup
	var events []watch.Event
	for {
		if initialEventsDone {
			var delay time.Duration
			switch {
			case !hiccupDone && len(events) > 0:
				delay = cfg.HiccupDuration
				hiccupDone = true
			case mode == WatcherSlow:
				delay = cfg.SlowDelay
			}
			if delay > 0 {
				select {
				case <-ctx.Done():
					return correctness.WatchResponse{Events: events, Err: ctx.Err()}
				case <-timer.C:
					return correctness.WatchResponse{Events: events}
				case <-time.After(delay):
				}
			}
		}
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
			if event.Type == watch.Bookmark {
				initialEventsDone = true
			}
			if event.Type == watch.Error {
				_, open := <-w.ResultChan()
				if open {
					return correctness.WatchResponse{Events: events, Err: errors.New("watch channel was not closed after watch.Error")}
				}
				return correctness.WatchResponse{Events: events}
			}
			if cfg.MaxEvents > 0 && len(events) >= cfg.MaxEvents {
				return correctness.WatchResponse{Events: events}
			}
		}
	}
}

func validPod(namespace, name string) *api.Pod {
	gracePeriod := int64(30)
	enableServiceLinks := true
	nodeName := nodeNames[rand.Intn(len(nodeNames))]
	appLabel := appLabels[rand.Intn(len(appLabels))]
	var podLabels map[string]string
	if appLabel != "" {
		podLabels = map[string]string{"app": appLabel}
	}
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
			Labels:    podLabels,
		},
		Spec: api.PodSpec{
			NodeName:                      nodeName,
			RestartPolicy:                 api.RestartPolicyAlways,
			TerminationGracePeriodSeconds: &gracePeriod,
			DNSPolicy:                     api.DNSClusterFirst,
			SecurityContext:               &api.PodSecurityContext{},
			SchedulerName:                 "default-scheduler",
			EnableServiceLinks:            &enableServiceLinks,
		},
	}
}
