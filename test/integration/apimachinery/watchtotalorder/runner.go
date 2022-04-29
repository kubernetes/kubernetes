/*
Copyright 2022 The Kubernetes Authors.

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

package watchtotalorder

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	kerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	cachetools "k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/klog/v2"
)

func init() {
	var logLevel string

	flag.Set("alsologtostderr", fmt.Sprintf("%t", true))
	flag.StringVar(&logLevel, "logLevel", "3", "test")
	flag.Lookup("v").Value.Set(logLevel)
}

type Logger interface {
	Log(string)
	Fatal(format string, args ...interface{})
}

func Run(ctx context.Context, cfg *rest.Config, logger Logger, ns string, strict bool) error {
	logger.Log("creating clients")
	apiExtensionsClient, err := apiextensionsclient.NewForConfig(cfg)
	if err != nil {
		return err
	}

	dynamicClient, err := dynamic.NewForConfig(cfg)
	if err != nil {
		return err
	}

	logger.Log("setting up the CustomResourceDefinition")
	// Create CRD and wait for the resource to be recognized and available.
	crd := fixtures.NewRandomNameV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped)
	crd, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionsClient, dynamicClient)
	if err != nil {
		return fmt.Errorf("failed to create CustomResourceDefinition: %w", err)
	}

	defer func() {
		if err := fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionsClient); err != nil {
			logger.Log(fmt.Sprintf("failed to clean up CustomResourceDefinition: %v", err))
		}
	}()

	gvr := schema.GroupVersionResource{
		Group:    crd.Spec.Group,
		Version:  crd.Spec.Versions[0].Name,
		Resource: crd.Spec.Names.Plural,
	}
	crClient := dynamicClient.Resource(gvr).Namespace(ns)

	iterations := 100

	logger.Log("getting a starting resourceVersion")
	items, err := crClient.List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("failed to determine a starting resourceVersion: %w", err)
	}
	resourceVersion := items.GetResourceVersion()
	klog.V(3).Infof("starting at resourceVersion=%s", resourceVersion)

	logger.Log("seeding the namespace with objects")
	var resourceVersions []string
	interactor := interactor{
		gvr:    gvr,
		kind:   crd.Spec.Names.Kind,
		client: crClient,
	}
	for idx := 0; idx < iterations; idx++ {
		rv, err := interactor.do(ctx, strict)
		if err != nil {
			return fmt.Errorf("failed to interact with the API server: %w", err)
		}
		resourceVersions = append(resourceVersions, rv)
	}

	logger.Log("starting watchers to accumulate events")
	lock := &sync.RWMutex{}
	producerWg := sync.WaitGroup{}
	consumerWg := sync.WaitGroup{}
	var producerErrors []error
	errChan := make(chan error)
	consumerWg.Add(1)
	go func() {
		defer consumerWg.Done()
		for err := range errChan {
			lock.Lock()
			producerErrors = append(producerErrors, err)
			lock.Unlock()
		}
	}()

	var lastResourceVersion string
	observationCtx, observationCancel := context.WithCancel(ctx)
	observedEvents := make([][]string, 2*iterations)
	for idx := 0; idx < len(observedEvents); idx++ {
		specificCtx, specificCancel := context.WithCancel(observationCtx)
		events := make(chan string)
		consumerWg.Add(1)
		go func(idx int) {
			defer consumerWg.Done()
			for {
				var observation string
				var ok bool
				select {
				case <-time.After(100 * time.Millisecond): // TODO: we could use a sync.Cond and broadcast here, but the UX of that does not mix with channels and contexts well
					lock.Lock()
					if lastResourceVersion != "" {
						// if we know the resourceVersion at which we need to stop, we should check to see if we've
						// already witnessed it on this watch channel, and, if so, truncate the observed events there.
						// otherwise, carry on and try again next time
						stoppingPointIndex := -1
						for i := range observedEvents[idx] {
							if observedEvents[idx][i] == lastResourceVersion {
								stoppingPointIndex = i
								break
							}
						}
						if stoppingPointIndex != -1 {
							klog.V(3).Infof("watcher %d saw the stopping point, %d events in", idx, stoppingPointIndex)
							observedEvents[idx] = observedEvents[idx][0 : stoppingPointIndex+1]
							specificCancel() // we've already seen the stopping point
						}
						// we need to keep looking for the stopping point
					}
					lock.Unlock()
				case observation, ok = <-events:
					if !ok {
						return
					}
					lock.Lock()
					observedEvents[idx] = append(observedEvents[idx], observation)
					lock.Unlock()
				}
			}
		}(idx)
		producerWg.Add(1)
		go func(reconnectAfter int, errSink chan<- error) {
			defer producerWg.Done()
			defer close(events)
			if err := observe(specificCtx, crClient, events, resourceVersion, reconnectAfter, iterations*2); err != nil && !errors.Is(err, context.Canceled) {
				errSink <- fmt.Errorf("failed to watch for events: %w", err)
			}
		}(idx, errChan)
	}

	logger.Log("mutating the objects in the namespace concurrently with the watchers")
	producerWg.Add(1)
	func() { // use a closure here to enable defer while keeping this work synchronous, since we want to wait *after* this is done
		defer producerWg.Done()
		for idx := 0; idx < iterations-1; idx++ {
			rv, err := interactor.do(ctx, strict)
			if err != nil {
				errChan <- fmt.Errorf("failed to interact with the API server: %w", err)
				return
			}
			lock.Lock()
			resourceVersions = append(resourceVersions, rv)
			lock.Unlock()
		}

		// we need to issue a CREATE here to ensure we have a resourceVersion to watch for, as DELETE calls don't return one
		rv, err := interactor.create(ctx)
		if err != nil {
			errChan <- fmt.Errorf("failed to interact with the API server: %w", err)
			return
		}
		lock.Lock()
		resourceVersions = append(resourceVersions, rv)
		lastResourceVersion = rv
		klog.V(3).Infof("committed the final resourceVersion=%s", lastResourceVersion)
		lock.Unlock()
	}()

	logger.Log("waiting for all the watchers to see the events we expected")
	go func() {
		<-time.After(wait.ForeverTestTimeout)
		errChan <- errors.New("timed out waiting to see all events")
		observationCancel() // cancel the producing goroutines
	}()
	producerWg.Wait() // wait for producers to finish, so we can close errChan as no writers will exist for it
	close(errChan)    // close errChan to stop the remaining consumers
	consumerWg.Wait() // wait for all the consumers to complete
	lock.Lock()
	defer lock.Unlock()

	if len(producerErrors) != 0 {
		return kerrors.NewAggregate(producerErrors)
	}

	logger.Log("checking to see that all watchers saw all events, in the same order")
	reference := observedEvents[0] // this watcher never reconnected and is used as our reference
	unique := sets.NewString(reference...)
	if unique.Len() != len(reference) {
		return fmt.Errorf("the reference watcher saw duplicate events: %v", cmp.Diff(unique.List(), reference))
	}
	if strict {
		// we know our client created all events, but we don't know the resourceVersions of the DELETE events,
		// so the best we can do is ensure that the watch streams saw the correct number of events
		if len(reference) != len(resourceVersions) {
			return fmt.Errorf("the reference watcher did not see all %d client events: %v", len(resourceVersions), cmp.Diff(resourceVersions, reference))
		}
	}
	for i := 0; i < len(observedEvents); i++ {
		if diff := cmp.Diff(reference, observedEvents[i]); diff != "" {
			return fmt.Errorf("watcher %d observed a different set of events from the reference: %s", i, diff)
		}
	}

	logger.Log("checking to see that all watchers saw all events the test client produced, in the order they were produced")
	// note: we know at this point that all the event streams from every watcher are identical, so it is sufficient
	// to check the events observed by the first watcher for this invariant
	eventIdx := 0
	referenceIdx := 0
	for {
		if referenceIdx == len(resourceVersions) {
			break // we're done
		}

		if eventIdx == len(observedEvents[0]) {
			return fmt.Errorf("watcher 0 never observed event %d created by the test plan", referenceIdx)
		}

		if observedEvents[0][eventIdx] == resourceVersions[referenceIdx] || resourceVersions[referenceIdx] == "" {
			// either we found the event we were looking for, or we have an empty resource version due to a DELETE,
			// which we cannot search for; in either case, move on to the next reference value
			referenceIdx++
		}
		eventIdx++
	}

	return nil
}

const (
	createEvent = iota
	updateEvent
	deleteEvent
)

type interactor struct {
	gvr      schema.GroupVersionResource
	kind     string
	client   dynamic.ResourceInterface
	existing []*unstructured.Unstructured
	count    int
	updates  int
}

func (i *interactor) do(ctx context.Context, strict bool) (string, error) {
	var op int
	if len(i.existing) == 0 {
		op = createEvent
	} else {
		op = rand.Intn(3)
	}

	switch op {
	case createEvent:
		return i.create(ctx)
	case updateEvent:
		return i.update(ctx, strict)
	case deleteEvent:
		// clients cannot today learn the resourceVersion at which their DELETE call was committed
		return "", i.delete(ctx, strict)
	default:
		return "", fmt.Errorf("unknown operation %d", op)
	}
}

func (i *interactor) create(ctx context.Context) (string, error) {
	i.count++
	name := names.SimpleNameGenerator.GenerateName("cr")
	obj, err := i.client.Create(ctx, &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": i.gvr.Group + "/" + i.gvr.Version,
		"kind":       i.kind,
		"metadata": map[string]interface{}{
			"name": name,
			"labels": map[string]interface{}{
				"counter": "nil",
			},
		},
	}}, metav1.CreateOptions{})
	if err != nil {
		return "", err
	}
	i.existing = append(i.existing, obj)
	return obj.GetResourceVersion(), nil
}

func (i *interactor) update(ctx context.Context, strict bool) (string, error) {
	idx := rand.Intn(len(i.existing))
	key := i.existing[idx]
	counter := i.updates
	i.updates++
	var rvPatch string
	if strict {
		// if we're being strict, we can't allow some other client to interleave requests to this object
		rvPatch = fmt.Sprintf(`"resourceVersion":%q,`, key.GetResourceVersion())
	}
	obj, err := i.client.Patch(ctx, key.GetName(),
		types.MergePatchType, []byte(fmt.Sprintf(`{"metadata":{%s"labels":{"counter":"%d"}}}`, rvPatch, counter)),
		metav1.PatchOptions{},
	)
	if err != nil {
		return "", err
	}
	i.existing[idx] = obj
	return obj.GetResourceVersion(), nil
}

func (i *interactor) delete(ctx context.Context, strict bool) error {
	idx := rand.Intn(len(i.existing))
	key := i.existing[idx]
	i.existing = append(i.existing[:idx], i.existing[idx+1:]...)
	// the dynamic client (and all the typed clients) throw away the response body in the DELETE call, which we
	// need to determine the resource version at which we deleted the object, so we need to use the raw client
	var deleteOptions metav1.DeleteOptions
	if strict {
		// if we're being strict, we can't allow some other client to interleave requests to this object
		deleteOptions.Preconditions = metav1.NewRVDeletionPrecondition(key.GetResourceVersion()).Preconditions
	}
	return i.client.Delete(ctx, key.GetName(), deleteOptions)
}

func observe(ctx context.Context, client dynamic.ResourceInterface, sink chan<- string, fromResourceVersion string, reconnectAfter, abortAfter int) error {
	if reconnectAfter >= abortAfter {
		return fmt.Errorf("requested to reconnect after aborting: %d >= %d", reconnectAfter, abortAfter)
	}
	listWatcher := &cachetools.ListWatch{
		WatchFunc: func(listOptions metav1.ListOptions) (watch.Interface, error) {
			return client.Watch(ctx, listOptions)
		},
	}

	var endpoints []func() bool
	if reconnectAfter == 0 {
		// we have no intermediate reconnection, so we watch until we're cancelled
		endpoints = []func() bool{
			func() bool {
				return false
			},
		}
	} else {
		// we have an intermediate reconnection, after which we watch until we're cancelled
		var count int
		endpoints = []func() bool{
			func() bool {
				count++
				return count == reconnectAfter
			},
			func() bool {
				return false
			},
		}
	}

	for _, done := range endpoints {
		watcher, err := watchtools.NewRetryWatcher(fromResourceVersion, listWatcher)
		if err != nil {
			return err
		}

		highWater, err := observeEventsUntil(ctx, watcher, sink, done)
		watcher.Stop()
		if err != nil {
			return err
		}
		fromResourceVersion = highWater
		klog.V(3).Infof("watcher %d restarting at resourceVersion=%s", reconnectAfter, highWater)
	}
	return nil
}

func observeEventsUntil(ctx context.Context, watcher watch.Interface, sink chan<- string, done func() bool) (highWater string, err error) {
	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case evt, ok := <-watcher.ResultChan():
			if !ok {
				return "", nil
			}
			obj, ok := evt.Object.(metav1.Object)
			if !ok {
				return "", fmt.Errorf("expected a metav1.Object in the watch stream, got %T", evt.Object)
			}
			observedResourceVersion := obj.GetResourceVersion()
			select {
			case <-ctx.Done():
				return "", ctx.Err()
			case sink <- observedResourceVersion:
			}
			if done() {
				return observedResourceVersion, nil
			}
		}
	}
}
