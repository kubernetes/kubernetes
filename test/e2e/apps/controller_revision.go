/*
Copyright 2022 The Kubernetes Authors.

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

package apps

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/retry"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/controller"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edaemonset "k8s.io/kubernetes/test/e2e/framework/daemonset"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/pointer"
)

const (
	controllerRevisionRetryPeriod  = 1 * time.Second
	controllerRevisionRetryTimeout = 1 * time.Minute
)

// This test must be run in serial because it assumes the Daemon Set pods will
// always get scheduled.  If we run other tests in parallel, this may not
// happen.  In the future, running in parallel may work if we have an eviction
// model which lets the DS controller kick out other pods to make room.
// See https://issues.k8s.io/21767 for more details
var _ = SIGDescribe("ControllerRevision", framework.WithSerial(), func() {
	var f *framework.Framework

	ginkgo.AfterEach(func(ctx context.Context) {
		// Clean up
		daemonsets, err := f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "unable to dump DaemonSets")
		if daemonsets != nil && len(daemonsets.Items) > 0 {
			for _, ds := range daemonsets.Items {
				ginkgo.By(fmt.Sprintf("Deleting DaemonSet %q", ds.Name))
				framework.ExpectNoError(e2eresource.DeleteResourceAndWaitForGC(ctx, f.ClientSet, extensionsinternal.Kind("DaemonSet"), f.Namespace.Name, ds.Name))
				err = wait.PollUntilContextTimeout(ctx, dsRetryPeriod, dsRetryTimeout, true, checkRunningOnNoNodes(f, &ds))
				framework.ExpectNoError(err, "error waiting for daemon pod to be reaped")
			}
		}
		if daemonsets, err := f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).List(ctx, metav1.ListOptions{}); err == nil {
			framework.Logf("daemonset: %s", runtime.EncodeOrDie(scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...), daemonsets))
		} else {
			framework.Logf("unable to dump daemonsets: %v", err)
		}
		if pods, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{}); err == nil {
			framework.Logf("pods: %s", runtime.EncodeOrDie(scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...), pods))
		} else {
			framework.Logf("unable to dump pods: %v", err)
		}
		err = clearDaemonSetNodeLabels(ctx, f.ClientSet)
		framework.ExpectNoError(err)
	})

	f = framework.NewDefaultFramework("controllerrevisions")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	image := WebserverImage
	dsName := "e2e-" + utilrand.String(5) + "-daemon-set"

	var ns string
	var c clientset.Interface

	ginkgo.BeforeEach(func(ctx context.Context) {
		ns = f.Namespace.Name

		c = f.ClientSet

		updatedNS, err := patchNamespaceAnnotations(ctx, c, ns)
		framework.ExpectNoError(err)

		ns = updatedNS.Name

		err = clearDaemonSetNodeLabels(ctx, c)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.25
		Testname: ControllerRevision, resource lifecycle
		Description: Creating a DaemonSet MUST succeed. Listing all
		ControllerRevisions with a label selector MUST find only one.
		After patching the ControllerRevision with a new label, the label
		MUST be found. Creating a new ControllerRevision for the DaemonSet
		MUST succeed. Listing the ControllerRevisions by label selector
		MUST find only two. Deleting a ControllerRevision MUST succeed.
		Listing the ControllerRevisions by label selector MUST find only
		one. After updating the ControllerRevision with a new label, the
		label MUST be found. Patching the DaemonSet MUST succeed. Listing the
		ControllerRevisions by label selector MUST find only two. Deleting
		a collection of ControllerRevision via a label selector MUST succeed.
		Listing the ControllerRevisions by label selector MUST find only one.
		The current ControllerRevision revision MUST be 3.
	*/
	framework.ConformanceIt("should manage the lifecycle of a ControllerRevision", func(ctx context.Context) {
		csAppsV1 := f.ClientSet.AppsV1()

		dsLabel := map[string]string{"daemonset-name": dsName}
		dsLabelSelector := labels.SelectorFromSet(dsLabel).String()

		ginkgo.By(fmt.Sprintf("Creating DaemonSet %q", dsName))
		testDaemonset, err := csAppsV1.DaemonSets(ns).Create(ctx, newDaemonSetWithLabel(dsName, image, dsLabel), metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollUntilContextTimeout(ctx, dsRetryPeriod, dsRetryTimeout, true, checkRunningOnAllNodes(f, testDaemonset))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")
		err = e2edaemonset.CheckDaemonStatus(ctx, f, dsName)
		framework.ExpectNoError(err)

		ginkgo.By(fmt.Sprintf("Confirm DaemonSet %q successfully created with %q label", dsName, dsLabelSelector))
		dsList, err := csAppsV1.DaemonSets("").List(ctx, metav1.ListOptions{LabelSelector: dsLabelSelector})
		framework.ExpectNoError(err, "failed to list Daemon Sets")
		gomega.Expect(dsList.Items).To(gomega.HaveLen(1), "filtered list wasn't found")

		ds, err := c.AppsV1().DaemonSets(ns).Get(ctx, dsName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Listing across all namespaces to verify api endpoint: listAppsV1ControllerRevisionForAllNamespaces
		ginkgo.By(fmt.Sprintf("Listing all ControllerRevisions with label %q", dsLabelSelector))
		revs, err := csAppsV1.ControllerRevisions("").List(ctx, metav1.ListOptions{LabelSelector: dsLabelSelector})
		framework.ExpectNoError(err, "Failed to list ControllerRevision: %v", err)
		gomega.Expect(revs.Items).To(gomega.HaveLen(1), "Failed to find any controllerRevisions")

		// Locate the current ControllerRevision from the list
		var initialRevision *appsv1.ControllerRevision

		rev := revs.Items[0]
		oref := rev.OwnerReferences[0]
		if oref.Kind == "DaemonSet" && oref.UID == ds.UID {
			framework.Logf("Located ControllerRevision: %q", rev.Name)
			initialRevision, err = csAppsV1.ControllerRevisions(ns).Get(ctx, rev.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to lookup ControllerRevision: %v", err)
			gomega.Expect(initialRevision).NotTo(gomega.BeNil(), "failed to lookup ControllerRevision: %v", initialRevision)
		}

		ginkgo.By(fmt.Sprintf("Patching ControllerRevision %q", initialRevision.Name))
		payload := "{\"metadata\":{\"labels\":{\"" + initialRevision.Name + "\":\"patched\"}}}"
		patchedControllerRevision, err := csAppsV1.ControllerRevisions(ns).Patch(ctx, initialRevision.Name, types.StrategicMergePatchType, []byte(payload), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch ControllerRevision %s in namespace %s", initialRevision.Name, ns)
		gomega.Expect(patchedControllerRevision.Labels).To(gomega.HaveKeyWithValue(initialRevision.Name, "patched"), "Did not find 'patched' label for this ControllerRevision. Current labels: %v", patchedControllerRevision.Labels)
		framework.Logf("%s has been patched", patchedControllerRevision.Name)

		ginkgo.By("Create a new ControllerRevision")
		ds.Spec.Template.Spec.TerminationGracePeriodSeconds = pointer.Int64(1)
		newHash, newName := hashAndNameForDaemonSet(ds)
		newRevision := &appsv1.ControllerRevision{
			ObjectMeta: metav1.ObjectMeta{
				Name:            newName,
				Namespace:       ds.Namespace,
				Labels:          labelsutil.CloneAndAddLabel(ds.Spec.Template.Labels, appsv1.DefaultDaemonSetUniqueLabelKey, newHash),
				Annotations:     ds.Annotations,
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(ds, appsv1.SchemeGroupVersion.WithKind("DaemonSet"))},
			},
			Data:     initialRevision.Data,
			Revision: initialRevision.Revision + 1,
		}
		newControllerRevision, err := csAppsV1.ControllerRevisions(ns).Create(ctx, newRevision, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create ControllerRevision: %v", err)
		framework.Logf("Created ControllerRevision: %s", newControllerRevision.Name)

		ginkgo.By("Confirm that there are two ControllerRevisions")
		err = wait.PollUntilContextTimeout(ctx, controllerRevisionRetryPeriod, controllerRevisionRetryTimeout, true, checkControllerRevisionListQuantity(f, dsLabelSelector, 2))
		framework.ExpectNoError(err, "failed to count required ControllerRevisions")

		ginkgo.By(fmt.Sprintf("Deleting ControllerRevision %q", initialRevision.Name))
		err = csAppsV1.ControllerRevisions(ns).Delete(ctx, initialRevision.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Failed to delete ControllerRevision: %v", err)

		ginkgo.By("Confirm that there is only one ControllerRevision")
		err = wait.PollUntilContextTimeout(ctx, controllerRevisionRetryPeriod, controllerRevisionRetryTimeout, true, checkControllerRevisionListQuantity(f, dsLabelSelector, 1))
		framework.ExpectNoError(err, "failed to count required ControllerRevisions")

		listControllerRevisions, err := csAppsV1.ControllerRevisions(ns).List(ctx, metav1.ListOptions{})
		currentControllerRevision := listControllerRevisions.Items[0]

		ginkgo.By(fmt.Sprintf("Updating ControllerRevision %q", currentControllerRevision.Name))
		var updatedControllerRevision *appsv1.ControllerRevision

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			updatedControllerRevision, err = csAppsV1.ControllerRevisions(ns).Get(ctx, currentControllerRevision.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to get ControllerRevision %s", currentControllerRevision.Name)
			updatedControllerRevision.Labels[currentControllerRevision.Name] = "updated"
			updatedControllerRevision, err = csAppsV1.ControllerRevisions(ns).Update(ctx, updatedControllerRevision, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update ControllerRevision in namespace: %s", ns)
		gomega.Expect(updatedControllerRevision.Labels).To(gomega.HaveKeyWithValue(currentControllerRevision.Name, "updated"), "Did not find 'updated' label for this ControllerRevision. Current labels: %v", updatedControllerRevision.Labels)
		framework.Logf("%s has been updated", updatedControllerRevision.Name)

		ginkgo.By("Generate another ControllerRevision by patching the Daemonset")
		patch := fmt.Sprintf(`{"spec":{"template":{"spec":{"terminationGracePeriodSeconds": %d}}},"updateStrategy":{"type":"RollingUpdate"}}`, 1)

		_, err = c.AppsV1().DaemonSets(ns).Patch(ctx, dsName, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "error patching daemon set")

		ginkgo.By("Confirm that there are two ControllerRevisions")
		err = wait.PollUntilContextTimeout(ctx, controllerRevisionRetryPeriod, controllerRevisionRetryTimeout, true, checkControllerRevisionListQuantity(f, dsLabelSelector, 2))
		framework.ExpectNoError(err, "failed to count required ControllerRevisions")

		updatedLabel := map[string]string{updatedControllerRevision.Name: "updated"}
		updatedLabelSelector := labels.SelectorFromSet(updatedLabel).String()

		ginkgo.By(fmt.Sprintf("Removing a ControllerRevision via 'DeleteCollection' with labelSelector: %q", updatedLabelSelector))
		err = csAppsV1.ControllerRevisions(ns).DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: updatedLabelSelector})
		framework.ExpectNoError(err, "Failed to delete ControllerRevision: %v", err)

		ginkgo.By("Confirm that there is only one ControllerRevision")
		err = wait.PollUntilContextTimeout(ctx, controllerRevisionRetryPeriod, controllerRevisionRetryTimeout, true, checkControllerRevisionListQuantity(f, dsLabelSelector, 1))
		framework.ExpectNoError(err, "failed to count required ControllerRevisions")

		list, err := csAppsV1.ControllerRevisions(ns).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list ControllerRevision")
		gomega.Expect(list.Items[0].Revision).To(gomega.Equal(int64(3)), "failed to find the expected revision for the Controller")
		framework.Logf("ControllerRevision %q has revision %d", list.Items[0].Name, list.Items[0].Revision)
	})
})

func checkControllerRevisionListQuantity(f *framework.Framework, label string, quantity int) func(ctx context.Context) (bool, error) {
	return func(ctx context.Context) (bool, error) {
		var err error

		framework.Logf("Requesting list of ControllerRevisions to confirm quantity")

		list, err := f.ClientSet.AppsV1().ControllerRevisions(f.Namespace.Name).List(ctx, metav1.ListOptions{
			LabelSelector: label})
		if err != nil {
			return false, err
		}

		if len(list.Items) != quantity {
			return false, nil
		}
		framework.Logf("Found %d ControllerRevisions", quantity)
		return true, nil
	}
}

func hashAndNameForDaemonSet(ds *appsv1.DaemonSet) (string, string) {
	hash := fmt.Sprint(controller.ComputeHash(&ds.Spec.Template, ds.Status.CollisionCount))
	name := ds.Name + "-" + hash
	return hash, name
}
