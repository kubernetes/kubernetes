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

package e2e_federation

import (
	"fmt"

	. "github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"

	. "github.com/onsi/gomega"
)

const (
	FederationEventName = "federation-event"
)

// Create/delete event api objects.
var _ = framework.KubeDescribe("Federation events [Feature:Federation]", func() {
	f := fedframework.NewDefaultFederatedFramework("federation-event")

	Describe("Event objects [NoCluster]", func() {
		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			nsName := f.FederationNamespace.Name
			// Delete registered events.
			eventList, err := f.FederationClientset.Core().Events(nsName).List(metav1.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, event := range eventList.Items {
				err := f.FederationClientset.Core().Events(nsName).Delete(event.Name, &metav1.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred())
			}
		})

		It("should be created and deleted successfully", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)

			nsName := f.FederationNamespace.Name
			event := createEventOrFail(f.FederationClientset, nsName)
			By(fmt.Sprintf("Creation of event %q in namespace %q succeeded.  Deleting event.", event.Name, nsName))
			// Cleanup
			err := f.FederationClientset.Core().Events(nsName).Delete(event.Name, &metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting event %q in namespace %q", event.Name, event.Namespace)
			By(fmt.Sprintf("Deletion of event %q in namespace %q succeeded.", event.Name, nsName))
		})

	})
})

func createEventOrFail(clientset *federation_clientset.Clientset, namespace string) *v1.Event {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createEventOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federated event %q in namespace %q", FederationEventName, namespace))

	event := &v1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Name:      FederationEventName,
			Namespace: namespace,
		},
		InvolvedObject: v1.ObjectReference{
			Kind:       "Pod",
			Name:       "pod-name",
			Namespace:  namespace,
			UID:        "C934D34AFB20242",
			APIVersion: "version",
		},
		Source: v1.EventSource{
			Component: "kubelet",
			Host:      "kublet.node1",
		},
		Count: 1,
		Type:  v1.EventTypeNormal,
	}

	_, err := clientset.Core().Events(namespace).Create(event)
	framework.ExpectNoError(err, "Creating event %q in namespace %q", event.Name, namespace)
	By(fmt.Sprintf("Successfully created federated event %q in namespace %q", FederationEventName, namespace))
	return event
}
