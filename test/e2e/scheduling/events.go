/*
Copyright 2017 The Kubernetes Authors.

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

package scheduling

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo/v2"
)

func scheduleSuccessEvent(ns, podName, nodeName string) func(*v1.Event) bool {
	return func(e *v1.Event) bool {
		return e.Type == v1.EventTypeNormal &&
			e.Reason == "Scheduled" &&
			strings.HasPrefix(e.Name, podName) &&
			strings.Contains(e.Message, fmt.Sprintf("Successfully assigned %v/%v to %v", ns, podName, nodeName))
	}
}

func scheduleFailureEvent(podName string) func(*v1.Event) bool {
	return func(e *v1.Event) bool {
		return strings.HasPrefix(e.Name, podName) &&
			e.Type == "Warning" &&
			e.Reason == "FailedScheduling"
	}
}

// Action is a function to be performed by the system.
type Action func() error

// observeEventAfterAction returns true if an event matching the predicate was emitted
// from the system after performing the supplied action.
func observeEventAfterAction(c clientset.Interface, ns string, eventPredicate func(*v1.Event) bool, action Action) (bool, error) {
	observedMatchingEvent := false
	informerStartedChan := make(chan struct{})
	var informerStartedGuard sync.Once

	// Create an informer to list/watch events from the test framework namespace.
	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				ls, err := c.CoreV1().Events(ns).List(context.TODO(), options)
				return ls, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				// Signal parent goroutine that watching has begun.
				defer informerStartedGuard.Do(func() { close(informerStartedChan) })
				w, err := c.CoreV1().Events(ns).Watch(context.TODO(), options)
				return w, err
			},
		},
		&v1.Event{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				e, ok := obj.(*v1.Event)
				if !ok {
					framework.Failf("Expected *v1.Event, got %T %v", obj, obj)
				}
				ginkgo.By(fmt.Sprintf("Considering event: \nType = [%s], Name = [%s], Reason = [%s], Message = [%s]", e.Type, e.Name, e.Reason, e.Message))
				if eventPredicate(e) {
					observedMatchingEvent = true
				}
			},
		},
	)

	// Start the informer and block this goroutine waiting for the started signal.
	informerStopChan := make(chan struct{})
	defer func() { close(informerStopChan) }()
	go controller.Run(informerStopChan)
	<-informerStartedChan

	// Invoke the action function.
	err := action()
	if err != nil {
		return false, err
	}

	// Poll whether the informer has found a matching event with a timeout.
	// Wait up 2 minutes polling every second.
	timeout := 2 * time.Minute
	interval := 1 * time.Second
	err = wait.Poll(interval, timeout, func() (bool, error) {
		return observedMatchingEvent, nil
	})
	return err == nil, err
}
