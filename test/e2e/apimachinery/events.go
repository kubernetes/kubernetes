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

package apimachinery

import (
	"context"
	"encoding/json"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/types"
)

const (
	eventRetryPeriod  = 1 * time.Second
	eventRetryTimeout = 1 * time.Minute
)

var _ = ginkgo.Describe("[sig-api-machinery] Events", func() {
	f := framework.NewDefaultFramework("events")

	/*
			   Release: v1.19
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

	/*
	   Release: v1.19
	   Testname: Event, delete a collection
	   Description: A set of events is created with a label selector which MUST be found when listed.
	   The set of events is deleted and MUST NOT show up when listed by its label selector.
	*/
	framework.ConformanceIt("should delete a collection of events", func() {
		eventTestNames := []string{"test-event-1", "test-event-2", "test-event-3"}

		ginkgo.By("Create set of events")
		// create a test event in test namespace
		for _, eventTestName := range eventTestNames {
			eventMessage := "This is " + eventTestName
			_, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Create(context.TODO(), &v1.Event{

				ObjectMeta: metav1.ObjectMeta{
					Name:   eventTestName,
					Labels: map[string]string{"testevent-set": "true"},
				},
				Message: eventMessage,
				Reason:  "Test",
				Type:    "Normal",
				Count:   1,
				InvolvedObject: v1.ObjectReference{
					Namespace: f.Namespace.Name,
				},
			}, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create event")
			framework.Logf("created %v", eventTestName)
		}

		ginkgo.By("get a list of Events with a label in the current namespace")
		// get a list of events
		eventList, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{
			LabelSelector: "testevent-set=true",
		})
		framework.ExpectNoError(err, "failed to get a list of events")

		framework.ExpectEqual(len(eventList.Items), len(eventTestNames), "looking for expected number of pod templates events")

		ginkgo.By("delete collection of events")
		// delete collection

		framework.Logf("requesting DeleteCollection of events")
		err = f.ClientSet.CoreV1().Events(f.Namespace.Name).DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{
			LabelSelector: "testevent-set=true"})
		framework.ExpectNoError(err, "failed to delete the test event")

		ginkgo.By("check that the list of events matches the requested quantity")

		err = wait.PollImmediate(eventRetryPeriod, eventRetryTimeout, checkEventListQuantity(f, "testevent-set=true", 0))
		framework.ExpectNoError(err, "failed to count required events")
	})

})

func checkEventListQuantity(f *framework.Framework, label string, quantity int) func() (bool, error) {
	return func() (bool, error) {
		var err error

		framework.Logf("requesting list of events to confirm quantity")

		eventList, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{
			LabelSelector: label})

		if err != nil {
			return false, err
		}

		if len(eventList.Items) != quantity {
			return false, err
		}
		return true, nil
	}
}
