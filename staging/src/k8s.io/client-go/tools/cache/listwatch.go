/*
Copyright 2015 The Kubernetes Authors.

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

package cache

import (
	"time"

	"golang.org/x/net/context"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/pager"
)

// ListerWatcher is any object that knows how to perform an initial list and start a watch on a resource.
type ListerWatcher interface {
	// List should return a list type object; the Items field will be extracted, and the
	// ResourceVersion field will be used to start the watch in the right place.
	List(options metav1.ListOptions) (runtime.Object, error)
	// Watch should begin a watch at the specified version.
	Watch(options metav1.ListOptions) (watch.Interface, error)
}

// ListFunc knows how to list resources
type ListFunc func(options metav1.ListOptions) (runtime.Object, error)

// WatchFunc knows how to watch resources
type WatchFunc func(options metav1.ListOptions) (watch.Interface, error)

// ListWatch knows how to list and watch a set of apiserver resources.  It satisfies the ListerWatcher interface.
// It is a convenience function for users of NewReflector, etc.
// ListFunc and WatchFunc must not be nil
type ListWatch struct {
	ListFunc  ListFunc
	WatchFunc WatchFunc
	// DisableChunking requests no chunking for this list watcher.
	DisableChunking bool
}

// Getter interface knows how to access Get method from RESTClient.
type Getter interface {
	Get() *restclient.Request
}

// NewListWatchFromClient creates a new ListWatch from the specified client, resource, namespace and field selector.
func NewListWatchFromClient(c Getter, resource string, namespace string, fieldSelector fields.Selector) *ListWatch {
	listFunc := func(options metav1.ListOptions) (runtime.Object, error) {
		options.FieldSelector = fieldSelector.String()
		return c.Get().
			Namespace(namespace).
			Resource(resource).
			VersionedParams(&options, metav1.ParameterCodec).
			Do().
			Get()
	}
	watchFunc := func(options metav1.ListOptions) (watch.Interface, error) {
		options.Watch = true
		options.FieldSelector = fieldSelector.String()
		return c.Get().
			Namespace(namespace).
			Resource(resource).
			VersionedParams(&options, metav1.ParameterCodec).
			Watch()
	}
	return &ListWatch{ListFunc: listFunc, WatchFunc: watchFunc}
}

func timeoutFromListOptions(options metav1.ListOptions) time.Duration {
	if options.TimeoutSeconds != nil {
		return time.Duration(*options.TimeoutSeconds) * time.Second
	}
	return 0
}

// List a set of apiserver resources
func (lw *ListWatch) List(options metav1.ListOptions) (runtime.Object, error) {
	if !lw.DisableChunking {
		return pager.New(pager.SimplePageFunc(lw.ListFunc)).List(context.TODO(), options)
	}
	return lw.ListFunc(options)
}

// Watch a set of apiserver resources
func (lw *ListWatch) Watch(options metav1.ListOptions) (watch.Interface, error) {
	return lw.WatchFunc(options)
}

// ListWatchUntil checks the provided conditions against the items returned by the list watcher, returning wait.ErrWaitTimeout
// if timeout is exceeded without all conditions returning true, or an error if an error occurs.
// TODO: check for watch expired error and retry watch from latest point?  Same issue exists for Until.
func ListWatchUntil(timeout time.Duration, lw ListerWatcher, conditions ...watch.ConditionFunc) (*watch.Event, error) {
	if len(conditions) == 0 {
		return nil, nil
	}

	list, err := lw.List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	initialItems, err := meta.ExtractList(list)
	if err != nil {
		return nil, err
	}

	// use the initial items as simulated "adds"
	var lastEvent *watch.Event
	currIndex := 0
	passedConditions := 0
	for _, condition := range conditions {
		// check the next condition against the previous event and short circuit waiting for the next watch
		if lastEvent != nil {
			done, err := condition(*lastEvent)
			if err != nil {
				return lastEvent, err
			}
			if done {
				passedConditions = passedConditions + 1
				continue
			}
		}

	ConditionSucceeded:
		for currIndex < len(initialItems) {
			lastEvent = &watch.Event{Type: watch.Added, Object: initialItems[currIndex]}
			currIndex++

			done, err := condition(*lastEvent)
			if err != nil {
				return lastEvent, err
			}
			if done {
				passedConditions = passedConditions + 1
				break ConditionSucceeded
			}
		}
	}
	if passedConditions == len(conditions) {
		return lastEvent, nil
	}
	remainingConditions := conditions[passedConditions:]

	metaObj, err := meta.ListAccessor(list)
	if err != nil {
		return nil, err
	}
	currResourceVersion := metaObj.GetResourceVersion()

	watchInterface, err := lw.Watch(metav1.ListOptions{ResourceVersion: currResourceVersion})
	if err != nil {
		return nil, err
	}

	evt, err := watch.Until(timeout, watchInterface, remainingConditions...)
	if err == watch.ErrWatchClosed {
		// present a consistent error interface to callers
		err = wait.ErrWaitTimeout
	}
	return evt, err
}
