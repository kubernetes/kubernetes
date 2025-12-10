/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/retry"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

func extinguish(ctx context.Context, f *framework.Framework, totalNS int, maxAllowedAfterDel int, maxSeconds int) {
	ginkgo.By("Creating testing namespaces")
	wg := &sync.WaitGroup{}
	wg.Add(totalNS)
	for n := 0; n < totalNS; n++ {
		go func(n int) {
			defer wg.Done()
			defer ginkgo.GinkgoRecover()
			ns := fmt.Sprintf("nslifetest-%v", n)
			_, err := f.CreateNamespace(ctx, ns, nil)
			framework.ExpectNoError(err, "failed to create namespace: %s", ns)
		}(n)
	}
	wg.Wait()

	//Wait 10 seconds, then SEND delete requests for all the namespaces.
	ginkgo.By("Waiting 10 seconds")
	time.Sleep(10 * time.Second)
	deleteFilter := []string{"nslifetest"}
	deleted, err := framework.DeleteNamespaces(ctx, f.ClientSet, deleteFilter, nil /* skipFilter */)
	framework.ExpectNoError(err, "failed to delete namespace(s) containing: %s", deleteFilter)
	gomega.Expect(deleted).To(gomega.HaveLen(totalNS))

	ginkgo.By("Waiting for namespaces to vanish")
	//Now POLL until all namespaces have been eradicated.
	framework.ExpectNoError(wait.Poll(2*time.Second, time.Duration(maxSeconds)*time.Second,
		func() (bool, error) {
			var cnt = 0
			nsList, err := f.ClientSet.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
			if err != nil {
				return false, err
			}
			for _, item := range nsList.Items {
				if strings.Contains(item.Name, "nslifetest") {
					cnt++
				}
			}
			if cnt > maxAllowedAfterDel {
				framework.Logf("Remaining namespaces : %v", cnt)
				return false, nil
			}
			return true, nil
		}))
}

func ensurePodsAreRemovedWhenNamespaceIsDeleted(ctx context.Context, f *framework.Framework) {
	ginkgo.By("Creating a test namespace")
	namespaceName := "nsdeletetest"
	namespace, err := f.CreateNamespace(ctx, namespaceName, nil)
	framework.ExpectNoError(err, "failed to create namespace: %s", namespaceName)

	ginkgo.By("Waiting for a default service account to be provisioned in namespace")
	err = framework.WaitForDefaultServiceAccountInNamespace(ctx, f.ClientSet, namespace.Name)
	framework.ExpectNoError(err, "failure while waiting for a default service account to be provisioned in namespace: %s", namespace.Name)

	ginkgo.By("Creating a pod in the namespace")
	podName := "test-pod"
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "nginx",
					Image: imageutils.GetPauseImageName(),
				},
			},
		},
	}
	pod, err = f.ClientSet.CoreV1().Pods(namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create pod %s in namespace: %s", podName, namespace.Name)

	ginkgo.By("Waiting for the pod to have running status")
	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))

	ginkgo.By("Deleting the namespace")
	err = f.ClientSet.CoreV1().Namespaces().Delete(ctx, namespace.Name, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "failed to delete namespace: %s", namespace.Name)

	ginkgo.By("Waiting for the namespace to be removed.")
	maxWaitSeconds := int64(60) + *pod.Spec.TerminationGracePeriodSeconds
	framework.ExpectNoError(wait.Poll(1*time.Second, time.Duration(maxWaitSeconds)*time.Second,
		func() (bool, error) {
			_, err = f.ClientSet.CoreV1().Namespaces().Get(ctx, namespace.Name, metav1.GetOptions{})
			if err != nil && apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, nil
		}))

	ginkgo.By("Recreating the namespace")
	namespace, err = f.CreateNamespace(ctx, namespaceName, nil)
	framework.ExpectNoError(err, "failed to create namespace: %s", namespaceName)

	ginkgo.By("Verifying there are no pods in the namespace")
	_, err = f.ClientSet.CoreV1().Pods(namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
	gomega.Expect(err).To(gomega.HaveOccurred(), "failed to get pod %s in namespace: %s", pod.Name, namespace.Name)
}

func ensureServicesAreRemovedWhenNamespaceIsDeleted(ctx context.Context, f *framework.Framework) {
	var err error

	ginkgo.By("Creating a test namespace")
	namespaceName := "nsdeletetest"
	namespace, err := f.CreateNamespace(ctx, namespaceName, nil)
	framework.ExpectNoError(err, "failed to create namespace: %s", namespaceName)

	ginkgo.By("Waiting for a default service account to be provisioned in namespace")
	err = framework.WaitForDefaultServiceAccountInNamespace(ctx, f.ClientSet, namespace.Name)
	framework.ExpectNoError(err, "failure while waiting for a default service account to be provisioned in namespace: %s", namespace.Name)

	ginkgo.By("Creating a service in the namespace")
	serviceName := "test-service"
	labels := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: serviceName,
		},
		Spec: v1.ServiceSpec{
			Selector: labels,
			Ports: []v1.ServicePort{{
				Port:       80,
				TargetPort: intstr.FromInt32(80),
			}},
		},
	}
	service, err = f.ClientSet.CoreV1().Services(namespace.Name).Create(ctx, service, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create service %s in namespace %s", serviceName, namespace.Name)

	ginkgo.By("Deleting the namespace")
	err = f.ClientSet.CoreV1().Namespaces().Delete(ctx, namespace.Name, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "failed to delete namespace: %s", namespace.Name)

	ginkgo.By("Waiting for the namespace to be removed.")
	maxWaitSeconds := int64(60)
	framework.ExpectNoError(wait.Poll(1*time.Second, time.Duration(maxWaitSeconds)*time.Second,
		func() (bool, error) {
			_, err = f.ClientSet.CoreV1().Namespaces().Get(ctx, namespace.Name, metav1.GetOptions{})
			if err != nil && apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, nil
		}))

	ginkgo.By("Recreating the namespace")
	namespace, err = f.CreateNamespace(ctx, namespaceName, nil)
	framework.ExpectNoError(err, "failed to create namespace: %s", namespaceName)

	ginkgo.By("Verifying there is no service in the namespace")
	_, err = f.ClientSet.CoreV1().Services(namespace.Name).Get(ctx, service.Name, metav1.GetOptions{})
	gomega.Expect(err).To(gomega.HaveOccurred(), "failed to get service %s in namespace: %s", service.Name, namespace.Name)
}

// This test must run [Serial] due to the impact of running other parallel
// tests can have on its performance.  Each test that follows the common
// test framework follows this pattern:
//  1. Create a Namespace
//  2. Do work that generates content in that namespace
//  3. Delete a Namespace
//
// Creation of a Namespace is non-trivial since it requires waiting for a
// ServiceAccount to be generated.
// Deletion of a Namespace is non-trivial and performance intensive since
// its an orchestrated process.  The controller that handles deletion must
// query the namespace for all existing content, and then delete each piece
// of content in turn.  As the API surface grows to add more KIND objects
// that could exist in a Namespace, the number of calls that the namespace
// controller must orchestrate grows since it must LIST, DELETE (1x1) each
// KIND.
// There is work underway to improve this, but it's
// most likely not going to get significantly better until etcd v3.
// Going back to this test, this test generates 100 Namespace objects, and then
// rapidly deletes all of them.  This causes the NamespaceController to observe
// and attempt to process a large number of deletes concurrently.  In effect,
// it's like running 100 traditional e2e tests in parallel.  If the namespace
// controller orchestrating deletes is slowed down deleting another test's
// content then this test may fail.  Since the goal of this test is to soak
// Namespace creation, and soak Namespace deletion, its not appropriate to
// further soak the cluster with other parallel Namespace deletion activities
// that each have a variable amount of content in the associated Namespace.
// When run in [Serial] this test appears to delete Namespace objects at a
// rate of approximately 1 per second.
var _ = SIGDescribe("Namespaces", framework.WithSerial(), func() {

	f := framework.NewDefaultFramework("namespaces")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	/*
		Release: v1.11
		Testname: namespace-deletion-removes-pods
		Description: Ensure that if a namespace is deleted then all pods are removed from that namespace.
	*/
	framework.ConformanceIt("should ensure that all pods are removed when a namespace is deleted", func(ctx context.Context) {
		ensurePodsAreRemovedWhenNamespaceIsDeleted(ctx, f)
	})

	/*
		Release: v1.11
		Testname: namespace-deletion-removes-services
		Description: Ensure that if a namespace is deleted then all services are removed from that namespace.
	*/
	framework.ConformanceIt("should ensure that all services are removed when a namespace is deleted", func(ctx context.Context) {
		ensureServicesAreRemovedWhenNamespaceIsDeleted(ctx, f)
	})

	ginkgo.It("should delete fast enough (90 percent of 100 namespaces in 150 seconds)", func(ctx context.Context) {
		extinguish(ctx, f, 100, 10, 150)
	})

	// On hold until etcd3; see #7372
	f.It("should always delete fast (ALL of 100 namespaces in 150 seconds)", feature.ComprehensiveNamespaceDraining, func(ctx context.Context) {
		extinguish(ctx, f, 100, 0, 150)
	})

	/*
	   Release: v1.18
	   Testname: Namespace patching
	   Description: A Namespace is created.
	   The Namespace is patched.
	   The Namespace and MUST now include the new Label.
	*/
	framework.ConformanceIt("should patch a Namespace", func(ctx context.Context) {
		ginkgo.By("creating a Namespace")
		namespaceName := "nspatchtest-" + string(uuid.NewUUID())
		ns, err := f.CreateNamespace(ctx, namespaceName, nil)
		framework.ExpectNoError(err, "failed creating Namespace")
		namespaceName = ns.ObjectMeta.Name
		gomega.Expect(ns).To(apimachineryutils.HaveValidResourceVersion())

		ginkgo.By("patching the Namespace")
		nspatch, err := json.Marshal(map[string]interface{}{
			"metadata": map[string]interface{}{
				"labels": map[string]string{"testLabel": "testValue"},
			},
		})
		framework.ExpectNoError(err, "failed to marshal JSON patch data")
		patchedNS, err := f.ClientSet.CoreV1().Namespaces().Patch(ctx, namespaceName, types.StrategicMergePatchType, nspatch, metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch Namespace")
		gomega.Expect(resourceversion.CompareResourceVersion(ns.ResourceVersion, patchedNS.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ginkgo.By("get the Namespace and ensuring it has the label")
		namespace, err := f.ClientSet.CoreV1().Namespaces().Get(ctx, namespaceName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get Namespace")
		gomega.Expect(namespace.ObjectMeta.Labels).To(gomega.HaveKeyWithValue("testLabel", "testValue"), "namespace not patched")
	})

	/*
		Release: v1.25
		Testname: Namespace, apply changes to a namespace status
		Description: Getting the current namespace status MUST succeed. The reported status
		phase MUST be active. Given the patching of the namespace status, the fields MUST
		equal the new values. Given the updating of the namespace status, the fields MUST
		equal the new values.
	*/
	framework.ConformanceIt("should apply changes to a namespace status", func(ctx context.Context) {
		ns := f.Namespace.Name
		dc := f.DynamicClient
		nsResource := v1.SchemeGroupVersion.WithResource("namespaces")
		nsClient := f.ClientSet.CoreV1().Namespaces()

		ginkgo.By("Read namespace status")

		unstruct, err := dc.Resource(nsResource).Get(ctx, ns, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err, "failed to fetch NamespaceStatus %s", ns)
		nsStatus, err := unstructuredToNamespace(unstruct)
		framework.ExpectNoError(err, "Getting the status of the namespace %s", ns)
		gomega.Expect(nsStatus.Status.Phase).To(gomega.Equal(v1.NamespaceActive), "The phase returned was %v", nsStatus.Status.Phase)
		framework.Logf("Status: %#v", nsStatus.Status)

		ginkgo.By("Patch namespace status")

		nsCondition := v1.NamespaceCondition{
			Type:    "StatusPatch",
			Status:  v1.ConditionTrue,
			Reason:  "E2E",
			Message: "Patched by an e2e test",
		}
		nsConditionJSON, err := json.Marshal(nsCondition)
		framework.ExpectNoError(err, "failed to marshal namespace condition")

		patchedStatus, err := nsClient.Patch(ctx, ns, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"e2e-patched-ns-status":"`+ns+`"}},"status":{"conditions":[`+string(nsConditionJSON)+`]}}`),
			metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err, "Failed to patch status. err: %v ", err)
		gomega.Expect(patchedStatus.Annotations).To(gomega.HaveKeyWithValue("e2e-patched-ns-status", ns), "patched object should have the applied annotation")
		gomega.Expect(string(patchedStatus.Status.Conditions[len(patchedStatus.Status.Conditions)-1].Reason)).To(gomega.Equal("E2E"), "The Reason returned was %v", patchedStatus.Status.Conditions[0].Reason)
		gomega.Expect(string(patchedStatus.Status.Conditions[len(patchedStatus.Status.Conditions)-1].Message)).To(gomega.Equal("Patched by an e2e test"), "The Message returned was %v", patchedStatus.Status.Conditions[0].Reason)
		framework.Logf("Status.Condition: %#v", patchedStatus.Status.Conditions[len(patchedStatus.Status.Conditions)-1])

		ginkgo.By("Update namespace status")
		var statusUpdated *v1.Namespace

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			unstruct, err := dc.Resource(nsResource).Get(ctx, ns, metav1.GetOptions{}, "status")
			framework.ExpectNoError(err, "failed to fetch NamespaceStatus %s", ns)
			statusToUpdate, err := unstructuredToNamespace(unstruct)
			framework.ExpectNoError(err, "Getting the status of the namespace %s", ns)

			statusToUpdate.Status.Conditions = append(statusToUpdate.Status.Conditions, v1.NamespaceCondition{
				Type:    "StatusUpdate",
				Status:  v1.ConditionTrue,
				Reason:  "E2E",
				Message: "Updated by an e2e test",
			})
			statusUpdated, err = nsClient.UpdateStatus(ctx, statusToUpdate, metav1.UpdateOptions{})

			return err
		})
		framework.ExpectNoError(err, "failed to update namespace status %s", ns)
		gomega.Expect(statusUpdated.Status.Conditions).To(gomega.HaveLen(len(statusUpdated.Status.Conditions)), "updated object should have the applied condition, got %#v", statusUpdated.Status.Conditions)
		gomega.Expect(statusUpdated.Status.Conditions[len(statusUpdated.Status.Conditions)-1].Type).To(gomega.Equal(v1.NamespaceConditionType("StatusUpdate")), "updated object should have the approved condition, got %#v", statusUpdated.Status.Conditions)
		gomega.Expect(statusUpdated.Status.Conditions[len(statusUpdated.Status.Conditions)-1].Message).To(gomega.Equal("Updated by an e2e test"), "The Message returned was %v", statusUpdated.Status.Conditions[0].Message)
		framework.Logf("Status.Condition: %#v", statusUpdated.Status.Conditions[len(statusUpdated.Status.Conditions)-1])
	})

	/*
		Release: v1.26
		Testname: Namespace, apply update to a namespace
		Description: When updating the namespace it MUST
		succeed and the field MUST equal the new value.
	*/
	framework.ConformanceIt("should apply an update to a Namespace", func(ctx context.Context) {
		var err error
		var updatedNamespace *v1.Namespace
		ns := f.Namespace.Name
		cs := f.ClientSet

		ginkgo.By(fmt.Sprintf("Updating Namespace %q", ns))
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			updatedNamespace, err = cs.CoreV1().Namespaces().Get(ctx, ns, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to get Namespace %q", ns)

			updatedNamespace.Labels[ns] = "updated"
			updatedNamespace, err = cs.CoreV1().Namespaces().Update(ctx, updatedNamespace, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update Namespace: %q", ns)
		gomega.Expect(updatedNamespace.ObjectMeta.Labels).To(gomega.HaveKeyWithValue(ns, "updated"), "Failed to update namespace %q. Current Labels: %#v", ns, updatedNamespace.Labels)
		framework.Logf("Namespace %q now has labels, %#v", ns, updatedNamespace.Labels)
	})

	/*
		Release: v1.26
		Testname: Namespace, apply finalizer to a namespace
		Description: Attempt to create a Namespace which MUST be succeed.
		Updating the namespace with a fake finalizer MUST succeed. The
		fake finalizer MUST be found. Removing the fake finalizer from
		the namespace MUST succeed and MUST NOT be found.
	*/
	framework.ConformanceIt("should apply a finalizer to a Namespace", func(ctx context.Context) {

		fakeFinalizer := v1.FinalizerName("e2e.example.com/fakeFinalizer")
		var updatedNamespace *v1.Namespace
		nsName := "e2e-ns-" + utilrand.String(5)

		ginkgo.By(fmt.Sprintf("Creating namespace %q", nsName))
		testNamespace, err := f.CreateNamespace(ctx, nsName, nil)
		framework.ExpectNoError(err, "failed creating Namespace")
		ns := testNamespace.ObjectMeta.Name
		nsClient := f.ClientSet.CoreV1().Namespaces()
		framework.Logf("Namespace %q has %#v", testNamespace.Name, testNamespace.Spec.Finalizers)

		ginkgo.By(fmt.Sprintf("Adding e2e finalizer to namespace %q", ns))
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			updateNamespace, err := nsClient.Get(ctx, ns, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to get Namespace %q", ns)

			updateNamespace.Spec.Finalizers = append(updateNamespace.Spec.Finalizers, fakeFinalizer)
			updatedNamespace, err = nsClient.Finalize(ctx, updateNamespace, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to add finalizer to the namespace: %q", ns)

		var foundFinalizer bool
		for _, item := range updatedNamespace.Spec.Finalizers {
			if item == fakeFinalizer {
				foundFinalizer = true
				break
			}
		}
		if !foundFinalizer {
			framework.Failf("Finalizer %q was not found. Namespace %q has %#v", fakeFinalizer, updatedNamespace.Name, updatedNamespace.Spec.Finalizers)
		}
		framework.Logf("Namespace %q has %#v", updatedNamespace.Name, updatedNamespace.Spec.Finalizers)

		ginkgo.By(fmt.Sprintf("Removing e2e finalizer from namespace %q", ns))
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			updatedNamespace, err = nsClient.Get(ctx, ns, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to get namespace %q", ns)

			var finalizerList []v1.FinalizerName
			for _, item := range updatedNamespace.Spec.Finalizers {
				if item != fakeFinalizer {
					finalizerList = append(finalizerList, item)
				}
			}
			updatedNamespace.Spec.Finalizers = finalizerList
			updatedNamespace, err = nsClient.Finalize(ctx, updatedNamespace, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to remove finalizer from namespace: %q", ns)

		foundFinalizer = false
		for _, item := range updatedNamespace.Spec.Finalizers {
			if item == fakeFinalizer {
				foundFinalizer = true
				break
			}
		}
		if foundFinalizer {
			framework.Failf("Finalizer %q was found. Namespace %q has %#v", fakeFinalizer, updatedNamespace.Name, updatedNamespace.Spec.Finalizers)
		}
		framework.Logf("Namespace %q has %#v", updatedNamespace.Name, updatedNamespace.Spec.Finalizers)
	})

})

func unstructuredToNamespace(obj *unstructured.Unstructured) (*v1.Namespace, error) {
	json, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}
	ns := &v1.Namespace{}
	err = runtime.DecodeInto(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), json, ns)

	return ns, err
}

var _ = SIGDescribe("OrderedNamespaceDeletion", func() {
	f := framework.NewDefaultFramework("namespacedeletion")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	/*
		Release : v1.34
		Testname: Ordered Namespace Deletion
		Description: Pods must be deleted before other objects when deleting a namespace. See https://kep.k8s.io/5080
	*/
	f.It("namespace deletion should delete pod first", framework.WithConformance(), func(ctx context.Context) {
		ensurePodsAreRemovedFirstInOrderedNamespaceDeletion(ctx, f)
	})
})

func ensurePodsAreRemovedFirstInOrderedNamespaceDeletion(ctx context.Context, f *framework.Framework) {
	ginkgo.By("Creating a test namespace")
	namespaceName := "nsdeletetest"
	namespace, err := f.CreateNamespace(ctx, namespaceName, nil)
	framework.ExpectNoError(err, "failed to create namespace: %s", namespaceName)
	nsName := namespace.Name

	ginkgo.By("Waiting for a default service account to be provisioned in namespace")
	err = framework.WaitForDefaultServiceAccountInNamespace(ctx, f.ClientSet, nsName)
	framework.ExpectNoError(err, "failure while waiting for a default service account to be provisioned in namespace: %s", nsName)

	ginkgo.By("Creating a pod with finalizer in the namespace")
	podName := "test-pod"
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
			Finalizers: []string{
				"e2e.example.com/finalizer",
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "nginx",
					Image: imageutils.GetPauseImageName(),
				},
			},
		},
	}
	pod, err = f.ClientSet.CoreV1().Pods(nsName).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create pod %s in namespace: %s", podName, nsName)

	ginkgo.By("Waiting for the pod to have running status")
	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod))

	configMapName := "test-configmap"
	ginkgo.By(fmt.Sprintf("Creating a configmap %q in namespace %q", configMapName, nsName))
	configMap := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: nsName,
		},
		Data: map[string]string{
			"key": "value",
		},
	}
	_, err = f.ClientSet.CoreV1().ConfigMaps(nsName).Create(ctx, configMap, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create configmap %q in namespace %q", configMapName, nsName)

	ginkgo.By("Deleting the namespace")
	err = f.ClientSet.CoreV1().Namespaces().Delete(ctx, nsName, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "failed to delete namespace: %s", nsName)

	ginkgo.By("wait until namespace controller had time to process")
	err = wait.PollUntilContextTimeout(ctx, 2*time.Second, framework.DefaultNamespaceDeletionTimeout, true,
		func(ctx context.Context) (bool, error) {
			ns, err := f.ClientSet.CoreV1().Namespaces().Get(ctx, nsName, metav1.GetOptions{})
			if err != nil {
				if apierrors.IsNotFound(err) {
					return false, fmt.Errorf("namespace %s was deleted unexpectedly", nsName)
				}
				framework.Logf("Failed to get namespace %q: %v", nsName, err)
				return false, nil
			}
			if ns.Status.Phase != v1.NamespaceTerminating {
				framework.Logf("Namespace %q is not in Terminating phase, retrying...", nsName)
				return false, nil
			}
			hasContextFailure := false
			for _, cond := range ns.Status.Conditions {
				if cond.Type == v1.NamespaceDeletionContentFailure {
					hasContextFailure = true
				}
			}
			if !hasContextFailure {
				framework.Logf("Namespace %q does not yet have a NamespaceDeletionContentFailure condition, retrying...", nsName)
				return false, nil
			}
			return true, nil
		},
	)
	if err != nil {
		framework.Failf("namespace %s has not been processed by namespace controller: %v", nsName, err)
	}
	ginkgo.By("the pod should be deleted before processing deletion for other resources")
	err = wait.PollUntilContextTimeout(ctx, 2*time.Second, framework.DefaultNamespaceDeletionTimeout, true,
		func(ctx context.Context) (bool, error) {
			_, err = f.ClientSet.CoreV1().ConfigMaps(nsName).Get(ctx, configMapName, metav1.GetOptions{})
			if err != nil {
				if apierrors.IsNotFound(err) {
					return false, fmt.Errorf("configmap %q should still exist in namespace %q", configMapName, nsName)
				}
				framework.Logf("Failed to get configmap %q in namespace: %q: %v", configMapName, nsName, err)
				return false, nil
			}
			// the pod should exist and has a deletionTimestamp set
			pod, err = f.ClientSet.CoreV1().Pods(nsName).Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				if apierrors.IsNotFound(err) {
					return false, fmt.Errorf("failed to get pod %q in namespace %q", pod.Name, nsName)
				}
				framework.Logf("Failed to get pod %q in namespace: %q: %v", pod.Name, nsName, err)
				return false, nil
			}
			if pod.DeletionTimestamp == nil {
				framework.Logf("Pod %q in namespace %q does not yet have a metadata.deletionTimestamp set, retrying...", pod.Name, nsName)
				return false, nil
			}
			return true, nil
		},
	)
	if err != nil {
		framework.Failf("pod %s was not deleted before the configmap %s: %v", pod.Name, configMapName, err)
	}

	ginkgo.By(fmt.Sprintf("Removing finalizer from pod %q in namespace %q", podName, nsName))
	err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
		pod, err = f.ClientSet.CoreV1().Pods(nsName).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		pod.Finalizers = []string{}
		_, err = f.ClientSet.CoreV1().Pods(nsName).Update(ctx, pod, metav1.UpdateOptions{})
		return err
	})
	framework.ExpectNoError(err, "failed to update pod %q and remove finalizer in namespace %q", podName, nsName)

	ginkgo.By("Waiting for the pod to not be present in the namespace")
	framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, podName, nsName, f.Timeouts.PodDelete))

	ginkgo.By("Waiting for the namespace to be removed.")
	framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, 1*time.Second, framework.DefaultNamespaceDeletionTimeout, true,
		func(ctx context.Context) (bool, error) {
			_, err = f.ClientSet.CoreV1().Namespaces().Get(ctx, namespace.Name, metav1.GetOptions{})
			if err != nil && apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, nil
		}))
}
