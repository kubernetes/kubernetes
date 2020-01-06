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

package events

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/types"
)

// Action is a function to be performed by the system.
type Action func() error

var _ = ginkgo.Describe("[sig-api-machinery] Events", func() {
	f := framework.NewDefaultFramework("events")

	ginkgo.It("should ensure that an event can be fetched, patched, deleted, and listed", func() {
		eventTestName := "event-test"

		ginkgo.By("creating a test event")
		// create a test event in test namespace
		_, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Create(&v1.Event{
			ObjectMeta: metav1.ObjectMeta{
				Name: eventTestName,
				Labels: map[string]string{
					"testevent-constant": "true",
				},
			},
			Message: "This is a test event",
			Reason:  "Test",
			Type:    "Normal",
			Count:   1,
			InvolvedObject: v1.ObjectReference{
				Namespace: f.Namespace.Name,
			},
		})
		framework.ExpectNoError(err, "failed to create test event")

		ginkgo.By("listing all events in all namespaces")
		// get a list of Events in all namespaces to ensure endpoint coverage
		eventsList, err := f.ClientSet.CoreV1().Events("").List(metav1.ListOptions{
			LabelSelector: "testevent-constant=true",
		})
		framework.ExpectNoError(err, "failed list all events")

		foundCreatedEvent := false
		var eventCreatedName string
		for _, val := range eventsList.Items {
			if val.ObjectMeta.Name == eventTestName && val.ObjectMeta.Namespace == f.Namespace.Name {
				foundCreatedEvent = true
				eventCreatedName = val.ObjectMeta.Name
				break
			}
		}
		framework.ExpectEqual(foundCreatedEvent, true, "unable to find the test event")

		ginkgo.By("patching the test event")
		// patch the event's message
		eventPatchMessage := "This is a test event - patched"
		eventPatch, err := json.Marshal(map[string]interface{}{
			"message": eventPatchMessage,
		})
		framework.ExpectNoError(err, "failed to marshal the patch JSON payload")

		_, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).Patch(eventTestName, types.StrategicMergePatchType, []byte(eventPatch))
		framework.ExpectNoError(err, "failed to patch the test event")

		ginkgo.By("fetching the test event")
		// get event by name
		event, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Get(eventCreatedName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch the test event")
		framework.ExpectEqual(event.Message, eventPatchMessage, "test event message does not match patch message")

		ginkgo.By("deleting the test event")
		// delete original event
		err = f.ClientSet.CoreV1().Events(f.Namespace.Name).Delete(eventCreatedName, &metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete the test event")

		ginkgo.By("listing all events in all namespaces")
		// get a list of Events list namespace
		eventsList, err = f.ClientSet.CoreV1().Events("").List(metav1.ListOptions{
			LabelSelector: "testevent-constant=true",
		})
		framework.ExpectNoError(err, "fail to list all events")
		foundCreatedEvent = false
		for _, val := range eventsList.Items {
			if val.ObjectMeta.Name == eventTestName && val.ObjectMeta.Namespace == f.Namespace.Name {
				foundCreatedEvent = true
				break
			}
		}
		framework.ExpectEqual(foundCreatedEvent, false, "failed to find test event")
	})
})

// ObserveNodeUpdateAfterAction returns true if a node update matching the predicate was emitted
// from the system after performing the supplied action.
func ObserveNodeUpdateAfterAction(c clientset.Interface, nodeName string, nodePredicate func(*v1.Node) bool, action Action) (bool, error) {
	observedMatchingNode := false
	nodeSelector := fields.OneTermEqualSelector("metadata.name", nodeName)
	informerStartedChan := make(chan struct{})
	var informerStartedGuard sync.Once

	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.FieldSelector = nodeSelector.String()
				ls, err := c.CoreV1().Nodes().List(options)
				return ls, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				// Signal parent goroutine that watching has begun.
				defer informerStartedGuard.Do(func() { close(informerStartedChan) })
				options.FieldSelector = nodeSelector.String()
				w, err := c.CoreV1().Nodes().Watch(options)
				return w, err
			},
		},
		&v1.Node{},
		0,
		cache.ResourceEventHandlerFuncs{
			UpdateFunc: func(oldObj, newObj interface{}) {
				n, ok := newObj.(*v1.Node)
				framework.ExpectEqual(ok, true)
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

// ObserveEventAfterAction returns true if an event matching the predicate was emitted
// from the system after performing the supplied action.
func ObserveEventAfterAction(c clientset.Interface, ns string, eventPredicate func(*v1.Event) bool, action Action) (bool, error) {
	observedMatchingEvent := false
	informerStartedChan := make(chan struct{})
	var informerStartedGuard sync.Once

	// Create an informer to list/watch events from the test framework namespace.
	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				ls, err := c.CoreV1().Events(ns).List(options)
				return ls, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				// Signal parent goroutine that watching has begun.
				defer informerStartedGuard.Do(func() { close(informerStartedChan) })
				w, err := c.CoreV1().Events(ns).Watch(options)
				return w, err
			},
		},
		&v1.Event{},
		0,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				e, ok := obj.(*v1.Event)
				ginkgo.By(fmt.Sprintf("Considering event: \nType = [%s], Name = [%s], Reason = [%s], Message = [%s]", e.Type, e.Name, e.Reason, e.Message))
				framework.ExpectEqual(ok, true)
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

// WaitTimeoutForEvent waits the given timeout duration for an event to occur.
func WaitTimeoutForEvent(c clientset.Interface, namespace, eventSelector, msg string, timeout time.Duration) error {
	interval := 2 * time.Second
	return wait.PollImmediate(interval, timeout, eventOccurred(c, namespace, eventSelector, msg))
}

func eventOccurred(c clientset.Interface, namespace, eventSelector, msg string) wait.ConditionFunc {
	options := metav1.ListOptions{FieldSelector: eventSelector}
	return func() (bool, error) {
		events, err := c.CoreV1().Events(namespace).List(context.TODO(), options)
		if err != nil {
			return false, fmt.Errorf("got error while getting events: %v", err)
		}
		for _, event := range events.Items {
			if strings.Contains(event.Message, msg) {
				return true, nil
			}
		}
		return false, nil
	}
}
