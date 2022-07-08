/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	typedeventsv1 "k8s.io/client-go/kubernetes/typed/events/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/instrumentation/common"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"k8s.io/apimachinery/pkg/types"
)

func newTestEvent(namespace, name, label string) *eventsv1.Event {
	someTime := metav1.MicroTime{Time: time.Unix(1505828956, 0)}
	return &eventsv1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				label: "true",
			},
		},
		Regarding: v1.ObjectReference{
			Namespace: namespace,
		},
		EventTime:           someTime,
		Note:                "This is " + name,
		Action:              "Do",
		Reason:              "Test",
		Type:                "Normal",
		ReportingController: "test-controller",
		ReportingInstance:   "test-node",
	}
}

func eventExistsInList(client typedeventsv1.EventInterface, namespace, name string) bool {
	eventsList, err := client.List(context.TODO(), metav1.ListOptions{
		LabelSelector: "testevent-constant=true",
	})
	framework.ExpectNoError(err, "failed to list events")

	for _, val := range eventsList.Items {
		if val.ObjectMeta.Name == name && val.ObjectMeta.Namespace == namespace {
			return true
		}
	}
	return false
}

var _ = common.SIGDescribe("Events API", func() {
	f := framework.NewDefaultFramework("events")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var coreClient corev1.EventInterface
	var client typedeventsv1.EventInterface
	var clientAllNamespaces typedeventsv1.EventInterface

	ginkgo.BeforeEach(func() {
		coreClient = f.ClientSet.CoreV1().Events(f.Namespace.Name)
		client = f.ClientSet.EventsV1().Events(f.Namespace.Name)
		clientAllNamespaces = f.ClientSet.EventsV1().Events(metav1.NamespaceAll)
	})

	/*
		Release: v1.19
		Testname: New Event resource lifecycle, testing a single event
		Description: Create an event, the event MUST exist.
		The event is patched with a new note, the check MUST have the update note.
		The event is updated with a new series, the check MUST have the update series.
		The event is deleted and MUST NOT show up when listing all events.
	*/
	framework.ConformanceIt("should ensure that an event can be fetched, patched, deleted, and listed", func() {
		eventName := "event-test"

		ginkgo.By("creating a test event")
		_, err := client.Create(context.TODO(), newTestEvent(f.Namespace.Name, eventName, "testevent-constant"), metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create test event")

		ginkgo.By("listing events in all namespaces")
		foundCreatedEvent := eventExistsInList(clientAllNamespaces, f.Namespace.Name, eventName)
		if !foundCreatedEvent {
			framework.Failf("Failed to find test event %s in namespace %s, in list with cluster scope", eventName, f.Namespace.Name)
		}

		ginkgo.By("listing events in test namespace")
		foundCreatedEvent = eventExistsInList(client, f.Namespace.Name, eventName)
		if !foundCreatedEvent {
			framework.Failf("Failed to find test event %s in namespace %s, in list with namespace scope", eventName, f.Namespace.Name)
		}

		ginkgo.By("listing events with field selection filtering on source")
		filteredCoreV1List, err := coreClient.List(context.TODO(), metav1.ListOptions{FieldSelector: "source=test-controller"})
		framework.ExpectNoError(err, "failed to get filtered list")
		if len(filteredCoreV1List.Items) != 1 || filteredCoreV1List.Items[0].Name != eventName {
			framework.Failf("expected single event, got %#v", filteredCoreV1List.Items)
		}

		ginkgo.By("listing events with field selection filtering on reportingController")
		filteredEventsV1List, err := client.List(context.TODO(), metav1.ListOptions{FieldSelector: "reportingController=test-controller"})
		framework.ExpectNoError(err, "failed to get filtered list")
		if len(filteredEventsV1List.Items) != 1 || filteredEventsV1List.Items[0].Name != eventName {
			framework.Failf("expected single event, got %#v", filteredEventsV1List.Items)
		}

		ginkgo.By("getting the test event")
		testEvent, err := client.Get(context.TODO(), eventName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get test event")

		ginkgo.By("patching the test event")
		oldData, err := json.Marshal(testEvent)
		framework.ExpectNoError(err, "failed to marshal event")
		newEvent := testEvent.DeepCopy()
		eventSeries := &eventsv1.EventSeries{
			Count:            2,
			LastObservedTime: metav1.MicroTime{Time: time.Unix(1505828951, 0)},
		}
		newEvent.Series = eventSeries
		newData, err := json.Marshal(newEvent)
		framework.ExpectNoError(err, "failed to marshal new event")
		patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, eventsv1.Event{})
		framework.ExpectNoError(err, "failed to create two-way merge patch")

		_, err = client.Patch(context.TODO(), eventName, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch the test event")

		ginkgo.By("getting the test event")
		event, err := client.Get(context.TODO(), eventName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get test event")
		// clear ResourceVersion and ManagedFields which are set by control-plane
		event.ObjectMeta.ResourceVersion = ""
		testEvent.ObjectMeta.ResourceVersion = ""
		event.ObjectMeta.ManagedFields = nil
		testEvent.ObjectMeta.ManagedFields = nil

		testEvent.Series = eventSeries
		if !apiequality.Semantic.DeepEqual(testEvent, event) {
			framework.Failf("test event wasn't properly patched: %v", diff.ObjectReflectDiff(testEvent, event))
		}

		ginkgo.By("updating the test event")
		testEvent.Series = &eventsv1.EventSeries{
			Count:            100,
			LastObservedTime: metav1.MicroTime{Time: time.Unix(1505828956, 0)},
		}
		_, err = client.Update(context.TODO(), testEvent, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "failed to update the test event")

		ginkgo.By("getting the test event")
		event, err = client.Get(context.TODO(), eventName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get test event")
		// clear ResourceVersion and ManagedFields which are set by control-plane
		event.ObjectMeta.ResourceVersion = ""
		event.ObjectMeta.ManagedFields = nil
		if !apiequality.Semantic.DeepEqual(testEvent, event) {
			framework.Failf("test event wasn't properly updated: %v", diff.ObjectReflectDiff(testEvent, event))
		}

		ginkgo.By("deleting the test event")
		err = client.Delete(context.TODO(), eventName, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete the test event")

		ginkgo.By("listing events in all namespaces")
		foundCreatedEvent = eventExistsInList(clientAllNamespaces, f.Namespace.Name, eventName)
		if foundCreatedEvent {
			framework.Failf("Should not have found test event %s in namespace %s, in list with cluster scope after deletion", eventName, f.Namespace.Name)
		}

		ginkgo.By("listing events in test namespace")
		foundCreatedEvent = eventExistsInList(client, f.Namespace.Name, eventName)
		if foundCreatedEvent {
			framework.Failf("Should not have found test event %s in namespace %s, in list with namespace scope after deletion", eventName, f.Namespace.Name)
		}
	})

	/*
		Release: v1.19
		Testname: New Event resource lifecycle, testing a list of events
		Description: Create a list of events, the events MUST exist.
		The events are deleted and MUST NOT show up when listing all events.
	*/
	framework.ConformanceIt("should delete a collection of events", func() {
		eventNames := []string{"test-event-1", "test-event-2", "test-event-3"}

		ginkgo.By("Create set of events")
		for _, eventName := range eventNames {
			_, err := client.Create(context.TODO(), newTestEvent(f.Namespace.Name, eventName, "testevent-set"), metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create event")
		}

		ginkgo.By("get a list of Events with a label in the current namespace")
		eventList, err := client.List(context.TODO(), metav1.ListOptions{
			LabelSelector: "testevent-set=true",
		})
		framework.ExpectNoError(err, "failed to get a list of events")
		framework.ExpectEqual(len(eventList.Items), len(eventNames), fmt.Sprintf("unexpected event list: %#v", eventList))

		ginkgo.By("delete a list of events")
		framework.Logf("requesting DeleteCollection of events")
		err = client.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{
			LabelSelector: "testevent-set=true",
		})
		framework.ExpectNoError(err, "failed to delete the test event")

		ginkgo.By("check that the list of events matches the requested quantity")
		eventList, err = client.List(context.TODO(), metav1.ListOptions{
			LabelSelector: "testevent-set=true",
		})
		framework.ExpectNoError(err, "failed to get a list of events")
		framework.ExpectEqual(len(eventList.Items), 0, fmt.Sprintf("unexpected event list: %#v", eventList))
	})
})
