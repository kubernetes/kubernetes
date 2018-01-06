/*
Copyright 2016 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type Action func() error

// Returns true if a node update matching the predicate was emitted from the
// system after performing the supplied action.
func ObserveNodeUpdateAfterAction(f *framework.Framework, nodeName string, nodePredicate func(*v1.Node) bool, action Action) (bool, error) {
	observedMatchingNode := false
	nodeSelector := fields.OneTermEqualSelector("metadata.name", nodeName)
	informerStartedChan := make(chan struct{})
	var informerStartedGuard sync.Once

	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.FieldSelector = nodeSelector.String()
				ls, err := f.ClientSet.CoreV1().Nodes().List(options)
				return ls, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				// Signal parent goroutine that watching has begun.
				defer informerStartedGuard.Do(func() { close(informerStartedChan) })
				options.FieldSelector = nodeSelector.String()
				w, err := f.ClientSet.CoreV1().Nodes().Watch(options)
				return w, err
			},
		},
		&v1.Node{},
		0,
		cache.ResourceEventHandlerFuncs{
			UpdateFunc: func(oldObj, newObj interface{}) {
				n, ok := newObj.(*v1.Node)
				Expect(ok).To(Equal(true))
				if nodePredicate(n) {
					observedMatchingNode = true
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

	// Poll whether the informer has found a matching node update with a timeout.
	// Wait up 2 minutes polling every second.
	timeout := 2 * time.Minute
	interval := 1 * time.Second
	err = wait.Poll(interval, timeout, func() (bool, error) {
		return observedMatchingNode, nil
	})
	return err == nil, err
}

// Returns true if an event matching the predicate was emitted from the system
// after performing the supplied action.
func ObserveEventAfterAction(f *framework.Framework, eventPredicate func(*v1.Event) bool, action Action) (bool, error) {
	observedMatchingEvent := false
	informerStartedChan := make(chan struct{})
	var informerStartedGuard sync.Once

	// Create an informer to list/watch events from the test framework namespace.
	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				ls, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(options)
				return ls, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				// Signal parent goroutine that watching has begun.
				defer informerStartedGuard.Do(func() { close(informerStartedChan) })
				w, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Watch(options)
				return w, err
			},
		},
		&v1.Event{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				e, ok := obj.(*v1.Event)
				By(fmt.Sprintf("Considering event: \nType = [%s], Name = [%s], Reason = [%s], Message = [%s]", e.Type, e.Name, e.Reason, e.Message))
				Expect(ok).To(Equal(true))
				if ok && eventPredicate(e) {
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
