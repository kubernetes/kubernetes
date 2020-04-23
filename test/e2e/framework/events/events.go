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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/types"
)

// Action is a function to be performed by the system.
type Action func() error

var _ = ginkgo.Describe("[sig-api-machinery] Events", func() {
	f := framework.NewDefaultFramework("events")

	/*
			   Release : v1.19
			   Testname: Event resource lifecycle
			   Description: Create an event, the event MUST exist.
		           The event is patched with a new message, the check MUST have the update message.
		           The event is deleted and MUST NOT show up when listing all events.
	*/
	framework.ConformanceIt("should ensure that an event can be fetched, patched, deleted, and listed", func() {
		eventTestName := "event-test"

		ginkgo.By("creating a test event")
		// create a test event in test namespace
		_, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Create(context.TODO(), &v1.Event{
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
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create test event")

		ginkgo.By("listing all events in all namespaces")
		// get a list of Events in all namespaces to ensure endpoint coverage
		eventsList, err := f.ClientSet.CoreV1().Events("").List(context.TODO(), metav1.ListOptions{
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

		_, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).Patch(context.TODO(), eventTestName, types.StrategicMergePatchType, []byte(eventPatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch the test event")

		ginkgo.By("fetching the test event")
		// get event by name
		event, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Get(context.TODO(), eventCreatedName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch the test event")
		framework.ExpectEqual(event.Message, eventPatchMessage, "test event message does not match patch message")

		ginkgo.By("deleting the test event")
		// delete original event
		err = f.ClientSet.CoreV1().Events(f.Namespace.Name).Delete(context.TODO(), eventCreatedName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete the test event")

		ginkgo.By("listing all events in all namespaces")
		// get a list of Events list namespace
		eventsList, err = f.ClientSet.CoreV1().Events("").List(context.TODO(), metav1.ListOptions{
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
		framework.ExpectEqual(foundCreatedEvent, false, "should not have found test event after deletion")
	})
})

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
