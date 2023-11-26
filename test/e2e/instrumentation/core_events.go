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

package instrumentation

import (
	"context"
	"encoding/json"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/instrumentation/common"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/types"
)

const (
	eventRetryPeriod  = 1 * time.Second
	eventRetryTimeout = 1 * time.Minute
)

var _ = common.SIGDescribe("Events", func() {
	f := framework.NewDefaultFramework("events")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		Release: v1.25
		Testname: Event, manage lifecycle of an Event
		Description: Attempt to create an event which MUST succeed.
		Attempt to list all namespaces with a label selector which MUST
		succeed. One list MUST be found. The event is patched with a
		new message, the check MUST have the update message. The event
		is updated with a new series of events, the check MUST confirm
		this update. The event is deleted and MUST NOT show up when
		listing all events.
	*/
	framework.ConformanceIt("should manage the lifecycle of an event", func(ctx context.Context) {
		// As per SIG-Arch meeting 14 July 2022 this e2e test now supersede
		// e2e test "Event resource lifecycle", which has been removed.

		eventTestName := "event-test"

		ginkgo.By("creating a test event")
		// create a test event in test namespace
		_, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Create(ctx, &v1.Event{
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
		eventsList, err := f.ClientSet.CoreV1().Events("").List(ctx, metav1.ListOptions{
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
		if !foundCreatedEvent {
			framework.Failf("unable to find test event %s in namespace %s, full list of events is %+v", eventTestName, f.Namespace.Name, eventsList.Items)
		}

		ginkgo.By("patching the test event")
		// patch the event's message
		eventPatchMessage := "This is a test event - patched"
		eventPatch, err := json.Marshal(map[string]interface{}{
			"message": eventPatchMessage,
		})
		framework.ExpectNoError(err, "failed to marshal the patch JSON payload")

		_, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).Patch(ctx, eventTestName, types.StrategicMergePatchType, []byte(eventPatch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch the test event")

		ginkgo.By("fetching the test event")
		// get event by name
		event, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Get(ctx, eventCreatedName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to fetch the test event")
		gomega.Expect(event.Message).To(gomega.Equal(eventPatchMessage), "test event message does not match patch message")

		ginkgo.By("updating the test event")

		testEvent, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Get(ctx, event.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get test event")

		testEvent.Series = &v1.EventSeries{
			Count:            100,
			LastObservedTime: metav1.MicroTime{Time: time.Unix(1505828956, 0)},
		}

		// clear ResourceVersion and ManagedFields which are set by control-plane
		testEvent.ObjectMeta.ResourceVersion = ""
		testEvent.ObjectMeta.ManagedFields = nil

		_, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).Update(ctx, testEvent, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "failed to update the test event")

		ginkgo.By("getting the test event")
		event, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).Get(ctx, testEvent.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get test event")
		// clear ResourceVersion and ManagedFields which are set by control-plane
		event.ObjectMeta.ResourceVersion = ""
		event.ObjectMeta.ManagedFields = nil
		if !apiequality.Semantic.DeepEqual(testEvent, event) {
			framework.Failf("test event wasn't properly updated: %v", cmp.Diff(testEvent, event))
		}

		ginkgo.By("deleting the test event")
		// delete original event
		err = f.ClientSet.CoreV1().Events(f.Namespace.Name).Delete(ctx, eventCreatedName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete the test event")

		ginkgo.By("listing all events in all namespaces")
		// get a list of Events list namespace
		eventsList, err = f.ClientSet.CoreV1().Events("").List(ctx, metav1.ListOptions{
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
		if foundCreatedEvent {
			framework.Failf("Should not have found test event %s in namespace %s, full list of events %+v", eventTestName, f.Namespace.Name, eventsList.Items)
		}
	})

	/*
	   Release: v1.20
	   Testname: Event, delete a collection
	   Description: A set of events is created with a label selector which MUST be found when listed.
	   The set of events is deleted and MUST NOT show up when listed by its label selector.
	*/
	framework.ConformanceIt("should delete a collection of events", func(ctx context.Context) {
		eventTestNames := []string{"test-event-1", "test-event-2", "test-event-3"}

		ginkgo.By("Create set of events")
		// create a test event in test namespace
		for _, eventTestName := range eventTestNames {
			eventMessage := "This is " + eventTestName
			_, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).Create(ctx, &v1.Event{

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
		eventList, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{
			LabelSelector: "testevent-set=true",
		})
		framework.ExpectNoError(err, "failed to get a list of events")

		gomega.Expect(eventList.Items).To(gomega.HaveLen(len(eventTestNames)), "looking for expected number of pod templates events")

		ginkgo.By("delete collection of events")
		// delete collection

		framework.Logf("requesting DeleteCollection of events")
		err = f.ClientSet.CoreV1().Events(f.Namespace.Name).DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{
			LabelSelector: "testevent-set=true"})
		framework.ExpectNoError(err, "failed to delete the test event")

		ginkgo.By("check that the list of events matches the requested quantity")

		err = wait.PollUntilContextTimeout(ctx, eventRetryPeriod, eventRetryTimeout, true, checkEventListQuantity(f, "testevent-set=true", 0))
		framework.ExpectNoError(err, "failed to count required events")
	})

})

func checkEventListQuantity(f *framework.Framework, label string, quantity int) func(ctx context.Context) (bool, error) {
	return func(ctx context.Context) (bool, error) {
		var err error

		framework.Logf("requesting list of events to confirm quantity")

		eventList, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{
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
