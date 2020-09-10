/*
Copyright 2015 The Kubernetes Authors.

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
	"reflect"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	testutils "k8s.io/kubernetes/test/utils"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	// this should not be a multiple of 5, because node status updates
	// every 5 seconds. See https://github.com/kubernetes/kubernetes/pull/14915.
	dsRetryPeriod  = 1 * time.Second
	dsRetryTimeout = 5 * time.Minute

	daemonsetLabelPrefix = "daemonset-"
	daemonsetNameLabel   = daemonsetLabelPrefix + "name"
	daemonsetColorLabel  = daemonsetLabelPrefix + "color"
)

// NamespaceNodeSelectors the annotation key scheduler.alpha.kubernetes.io/node-selector is for assigning
// node selectors labels to namespaces
var NamespaceNodeSelectors = []string{"scheduler.alpha.kubernetes.io/node-selector"}

type updateDSFunc func(*appsv1.DaemonSet)

// updateDaemonSetWithRetries updates daemonsets with the given applyUpdate func
// until it succeeds or a timeout expires.
func updateDaemonSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateDSFunc) (ds *appsv1.DaemonSet, err error) {
	daemonsets := c.AppsV1().DaemonSets(namespace)
	var updateErr error
	pollErr := wait.PollImmediate(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		if ds, err = daemonsets.Get(context.TODO(), name, metav1.GetOptions{}); err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(ds)
		if ds, err = daemonsets.Update(context.TODO(), ds, metav1.UpdateOptions{}); err == nil {
			framework.Logf("Updating DaemonSet %s", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("couldn't apply the provided updated to DaemonSet %q: %v", name, updateErr)
	}
	return ds, pollErr
}

// This test must be run in serial because it assumes the Daemon Set pods will
// always get scheduled.  If we run other tests in parallel, this may not
// happen.  In the future, running in parallel may work if we have an eviction
// model which lets the DS controller kick out other pods to make room.
// See http://issues.k8s.io/21767 for more details
var _ = SIGDescribe("Daemon set [Serial]", func() {
	var f *framework.Framework

	ginkgo.AfterEach(func() {
		// Clean up
		daemonsets, err := f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err, "unable to dump DaemonSets")
		if daemonsets != nil && len(daemonsets.Items) > 0 {
			for _, ds := range daemonsets.Items {
				ginkgo.By(fmt.Sprintf("Deleting DaemonSet %q", ds.Name))
				framework.ExpectNoError(e2eresource.DeleteResourceAndWaitForGC(f.ClientSet, extensionsinternal.Kind("DaemonSet"), f.Namespace.Name, ds.Name))
				err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, &ds))
				framework.ExpectNoError(err, "error waiting for daemon pod to be reaped")
			}
		}
		if daemonsets, err := f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{}); err == nil {
			framework.Logf("daemonset: %s", runtime.EncodeOrDie(scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...), daemonsets))
		} else {
			framework.Logf("unable to dump daemonsets: %v", err)
		}
		if pods, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{}); err == nil {
			framework.Logf("pods: %s", runtime.EncodeOrDie(scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...), pods))
		} else {
			framework.Logf("unable to dump pods: %v", err)
		}
		err = clearDaemonSetNodeLabels(f.ClientSet)
		framework.ExpectNoError(err)
	})

	f = framework.NewDefaultFramework("daemonsets")

	image := WebserverImage
	dsName := "daemon-set"

	var ns string
	var c clientset.Interface

	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name

		c = f.ClientSet

		updatedNS, err := updateNamespaceAnnotations(c, ns)
		framework.ExpectNoError(err)

		ns = updatedNS.Name

		err = clearDaemonSetNodeLabels(c)
		framework.ExpectNoError(err)
	})

	/*
	  Release: v1.10
	  Testname: DaemonSet-Creation
	  Description: A conformant Kubernetes distribution MUST support the creation of DaemonSets. When a DaemonSet
	  Pod is deleted, the DaemonSet controller MUST create a replacement Pod.
	*/
	framework.ConformanceIt("should run and stop simple daemon", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		ginkgo.By(fmt.Sprintf("Creating simple DaemonSet %q", dsName))
		ds, err := c.AppsV1().DaemonSets(ns).Create(context.TODO(), newDaemonSet(dsName, image, label), metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")
		err = checkDaemonStatus(f, dsName)
		framework.ExpectNoError(err)

		ginkgo.By("Stop a daemon pod, check that the daemon pod is revived.")
		podList := listDaemonPods(c, ns, label)
		pod := podList.Items[0]
		err = c.CoreV1().Pods(ns).Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to revive")
	})

	/*
	  Release: v1.10
	  Testname: DaemonSet-NodeSelection
	  Description: A conformant Kubernetes distribution MUST support DaemonSet Pod node selection via label
	  selectors.
	*/
	framework.ConformanceIt("should run and stop complex daemon", func() {
		complexLabel := map[string]string{daemonsetNameLabel: dsName}
		nodeSelector := map[string]string{daemonsetColorLabel: "blue"}
		framework.Logf("Creating daemon %q with a node selector", dsName)
		ds := newDaemonSet(dsName, image, complexLabel)
		ds.Spec.Template.Spec.NodeSelector = nodeSelector
		ds, err := c.AppsV1().DaemonSets(ns).Create(context.TODO(), ds, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Initially, daemon pods should not be running on any nodes.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pods to be running on no nodes")

		ginkgo.By("Change node label to blue, check that daemon pod is launched.")
		node, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
		framework.ExpectNoError(err)
		newNode, err := setDaemonSetNodeLabels(c, node.Name, nodeSelector)
		framework.ExpectNoError(err, "error setting labels on node")
		daemonSetLabels, _ := separateDaemonSetNodeLabels(newNode.Labels)
		framework.ExpectEqual(len(daemonSetLabels), 1)
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodOnNodes(f, ds, []string{newNode.Name}))
		framework.ExpectNoError(err, "error waiting for daemon pods to be running on new nodes")
		err = checkDaemonStatus(f, dsName)
		framework.ExpectNoError(err)

		ginkgo.By("Update the node label to green, and wait for daemons to be unscheduled")
		nodeSelector[daemonsetColorLabel] = "green"
		greenNode, err := setDaemonSetNodeLabels(c, node.Name, nodeSelector)
		framework.ExpectNoError(err, "error removing labels on node")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to not be running on nodes")

		ginkgo.By("Update DaemonSet node selector to green, and change its update strategy to RollingUpdate")
		patch := fmt.Sprintf(`{"spec":{"template":{"spec":{"nodeSelector":{"%s":"%s"}}},"updateStrategy":{"type":"RollingUpdate"}}}`,
			daemonsetColorLabel, greenNode.Labels[daemonsetColorLabel])
		ds, err = c.AppsV1().DaemonSets(ns).Patch(context.TODO(), dsName, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
		framework.ExpectNoError(err, "error patching daemon set")
		daemonSetLabels, _ = separateDaemonSetNodeLabels(greenNode.Labels)
		framework.ExpectEqual(len(daemonSetLabels), 1)
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodOnNodes(f, ds, []string{greenNode.Name}))
		framework.ExpectNoError(err, "error waiting for daemon pods to be running on new nodes")
		err = checkDaemonStatus(f, dsName)
		framework.ExpectNoError(err)
	})

	// We defer adding this test to conformance pending the disposition of moving DaemonSet scheduling logic to the
	// default scheduler.
	ginkgo.It("should run and stop complex daemon with node affinity", func() {
		complexLabel := map[string]string{daemonsetNameLabel: dsName}
		nodeSelector := map[string]string{daemonsetColorLabel: "blue"}
		framework.Logf("Creating daemon %q with a node affinity", dsName)
		ds := newDaemonSet(dsName, image, complexLabel)
		ds.Spec.Template.Spec.Affinity = &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      daemonsetColorLabel,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{nodeSelector[daemonsetColorLabel]},
								},
							},
						},
					},
				},
			},
		}
		ds, err := c.AppsV1().DaemonSets(ns).Create(context.TODO(), ds, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Initially, daemon pods should not be running on any nodes.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pods to be running on no nodes")

		ginkgo.By("Change node label to blue, check that daemon pod is launched.")
		node, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
		framework.ExpectNoError(err)
		newNode, err := setDaemonSetNodeLabels(c, node.Name, nodeSelector)
		framework.ExpectNoError(err, "error setting labels on node")
		daemonSetLabels, _ := separateDaemonSetNodeLabels(newNode.Labels)
		framework.ExpectEqual(len(daemonSetLabels), 1)
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodOnNodes(f, ds, []string{newNode.Name}))
		framework.ExpectNoError(err, "error waiting for daemon pods to be running on new nodes")
		err = checkDaemonStatus(f, dsName)
		framework.ExpectNoError(err)

		ginkgo.By("Remove the node label and wait for daemons to be unscheduled")
		_, err = setDaemonSetNodeLabels(c, node.Name, map[string]string{})
		framework.ExpectNoError(err, "error removing labels on node")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to not be running on nodes")
	})

	/*
	  Release: v1.10
	  Testname: DaemonSet-FailedPodCreation
	  Description: A conformant Kubernetes distribution MUST create new DaemonSet Pods when they fail.
	*/
	framework.ConformanceIt("should retry creating failed daemon pods", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		ginkgo.By(fmt.Sprintf("Creating a simple DaemonSet %q", dsName))
		ds, err := c.AppsV1().DaemonSets(ns).Create(context.TODO(), newDaemonSet(dsName, image, label), metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")
		err = checkDaemonStatus(f, dsName)
		framework.ExpectNoError(err)

		ginkgo.By("Set a daemon pod's phase to 'Failed', check that the daemon pod is revived.")
		podList := listDaemonPods(c, ns, label)
		pod := podList.Items[0]
		pod.ResourceVersion = ""
		pod.Status.Phase = v1.PodFailed
		_, err = c.CoreV1().Pods(ns).UpdateStatus(context.TODO(), &pod, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "error failing a daemon pod")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to revive")

		ginkgo.By("Wait for the failed daemon pod to be completely deleted.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, waitFailedDaemonPodDeleted(c, &pod))
		framework.ExpectNoError(err, "error waiting for the failed daemon pod to be completely deleted")
	})

	// This test should not be added to conformance. We will consider deprecating OnDelete when the
	// extensions/v1beta1 and apps/v1beta1 are removed.
	ginkgo.It("should not update pod when spec was updated and update strategy is OnDelete", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		framework.Logf("Creating simple daemon set %s", dsName)
		ds := newDaemonSet(dsName, image, label)
		ds.Spec.UpdateStrategy = appsv1.DaemonSetUpdateStrategy{Type: appsv1.OnDeleteDaemonSetStrategyType}
		ds, err := c.AppsV1().DaemonSets(ns).Create(context.TODO(), ds, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")

		// Check history and labels
		ds, err = c.AppsV1().DaemonSets(ns).Get(context.TODO(), ds.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		waitForHistoryCreated(c, ns, label, 1)
		first := curHistory(listDaemonHistories(c, ns, label), ds)
		firstHash := first.Labels[appsv1.DefaultDaemonSetUniqueLabelKey]
		framework.ExpectEqual(first.Revision, int64(1))
		checkDaemonSetPodsLabels(listDaemonPods(c, ns, label), firstHash)

		ginkgo.By("Update daemon pods image.")
		patch := getDaemonSetImagePatch(ds.Spec.Template.Spec.Containers[0].Name, AgnhostImage)
		ds, err = c.AppsV1().DaemonSets(ns).Patch(context.TODO(), dsName, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods images aren't updated.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodsImageAndAvailability(c, ds, image, 0))
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods are still running on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")

		// Check history and labels
		ds, err = c.AppsV1().DaemonSets(ns).Get(context.TODO(), ds.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		waitForHistoryCreated(c, ns, label, 2)
		cur := curHistory(listDaemonHistories(c, ns, label), ds)
		framework.ExpectEqual(cur.Revision, int64(2))
		framework.ExpectNotEqual(cur.Labels[appsv1.DefaultDaemonSetUniqueLabelKey], firstHash)
		checkDaemonSetPodsLabels(listDaemonPods(c, ns, label), firstHash)
	})

	/*
	  Release: v1.10
	  Testname: DaemonSet-RollingUpdate
	  Description: A conformant Kubernetes distribution MUST support DaemonSet RollingUpdates.
	*/
	framework.ConformanceIt("should update pod when spec was updated and update strategy is RollingUpdate", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		framework.Logf("Creating simple daemon set %s", dsName)
		ds := newDaemonSet(dsName, image, label)
		ds.Spec.UpdateStrategy = appsv1.DaemonSetUpdateStrategy{Type: appsv1.RollingUpdateDaemonSetStrategyType}
		ds, err := c.AppsV1().DaemonSets(ns).Create(context.TODO(), ds, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")

		// Check history and labels
		ds, err = c.AppsV1().DaemonSets(ns).Get(context.TODO(), ds.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		waitForHistoryCreated(c, ns, label, 1)
		cur := curHistory(listDaemonHistories(c, ns, label), ds)
		hash := cur.Labels[appsv1.DefaultDaemonSetUniqueLabelKey]
		framework.ExpectEqual(cur.Revision, int64(1))
		checkDaemonSetPodsLabels(listDaemonPods(c, ns, label), hash)

		ginkgo.By("Update daemon pods image.")
		patch := getDaemonSetImagePatch(ds.Spec.Template.Spec.Containers[0].Name, AgnhostImage)
		ds, err = c.AppsV1().DaemonSets(ns).Patch(context.TODO(), dsName, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
		framework.ExpectNoError(err)

		// Time to complete the rolling upgrade is proportional to the number of nodes in the cluster.
		// Get the number of nodes, and set the timeout appropriately.
		nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err)
		nodeCount := len(nodes.Items)
		retryTimeout := dsRetryTimeout + time.Duration(nodeCount*30)*time.Second

		ginkgo.By("Check that daemon pods images are updated.")
		err = wait.PollImmediate(dsRetryPeriod, retryTimeout, checkDaemonPodsImageAndAvailability(c, ds, AgnhostImage, 1))
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods are still running on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")

		// Check history and labels
		ds, err = c.AppsV1().DaemonSets(ns).Get(context.TODO(), ds.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		waitForHistoryCreated(c, ns, label, 2)
		cur = curHistory(listDaemonHistories(c, ns, label), ds)
		hash = cur.Labels[appsv1.DefaultDaemonSetUniqueLabelKey]
		framework.ExpectEqual(cur.Revision, int64(2))
		checkDaemonSetPodsLabels(listDaemonPods(c, ns, label), hash)
	})

	/*
	  Release: v1.10
	  Testname: DaemonSet-Rollback
	  Description: A conformant Kubernetes distribution MUST support automated, minimally disruptive
	  rollback of updates to a DaemonSet.
	*/
	framework.ConformanceIt("should rollback without unnecessary restarts", func() {
		schedulableNodes, err := e2enode.GetReadySchedulableNodes(c)
		framework.ExpectNoError(err)
		gomega.Expect(len(schedulableNodes.Items)).To(gomega.BeNumerically(">", 1), "Conformance test suite needs a cluster with at least 2 nodes.")
		framework.Logf("Create a RollingUpdate DaemonSet")
		label := map[string]string{daemonsetNameLabel: dsName}
		ds := newDaemonSet(dsName, image, label)
		ds.Spec.UpdateStrategy = appsv1.DaemonSetUpdateStrategy{Type: appsv1.RollingUpdateDaemonSetStrategyType}
		ds, err = c.AppsV1().DaemonSets(ns).Create(context.TODO(), ds, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		framework.Logf("Check that daemon pods launch on every node of the cluster")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")

		framework.Logf("Update the DaemonSet to trigger a rollout")
		// We use a nonexistent image here, so that we make sure it won't finish
		newImage := "foo:non-existent"
		newDS, err := updateDaemonSetWithRetries(c, ns, ds.Name, func(update *appsv1.DaemonSet) {
			update.Spec.Template.Spec.Containers[0].Image = newImage
		})
		framework.ExpectNoError(err)

		// Make sure we're in the middle of a rollout
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkAtLeastOneNewPod(c, ns, label, newImage))
		framework.ExpectNoError(err)

		pods := listDaemonPods(c, ns, label)
		var existingPods, newPods []*v1.Pod
		for i := range pods.Items {
			pod := pods.Items[i]
			image := pod.Spec.Containers[0].Image
			switch image {
			case ds.Spec.Template.Spec.Containers[0].Image:
				existingPods = append(existingPods, &pod)
			case newDS.Spec.Template.Spec.Containers[0].Image:
				newPods = append(newPods, &pod)
			default:
				framework.Failf("unexpected pod found, image = %s", image)
			}
		}
		schedulableNodes, err = e2enode.GetReadySchedulableNodes(c)
		framework.ExpectNoError(err)
		if len(schedulableNodes.Items) < 2 {
			framework.ExpectEqual(len(existingPods), 0)
		} else {
			framework.ExpectNotEqual(len(existingPods), 0)
		}
		framework.ExpectNotEqual(len(newPods), 0)

		framework.Logf("Roll back the DaemonSet before rollout is complete")
		rollbackDS, err := updateDaemonSetWithRetries(c, ns, ds.Name, func(update *appsv1.DaemonSet) {
			update.Spec.Template.Spec.Containers[0].Image = image
		})
		framework.ExpectNoError(err)

		framework.Logf("Make sure DaemonSet rollback is complete")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodsImageAndAvailability(c, rollbackDS, image, 1))
		framework.ExpectNoError(err)

		// After rollback is done, compare current pods with previous old pods during rollout, to make sure they're not restarted
		pods = listDaemonPods(c, ns, label)
		rollbackPods := map[string]bool{}
		for _, pod := range pods.Items {
			rollbackPods[pod.Name] = true
		}
		for _, pod := range existingPods {
			framework.ExpectEqual(rollbackPods[pod.Name], true, fmt.Sprintf("unexpected pod %s be restarted", pod.Name))
		}
	})
})

// getDaemonSetImagePatch generates a patch for updating a DaemonSet's container image
func getDaemonSetImagePatch(containerName, containerImage string) string {
	return fmt.Sprintf(`{"spec":{"template":{"spec":{"containers":[{"name":"%s","image":"%s"}]}}}}`, containerName, containerImage)
}

func newDaemonSet(dsName, image string, label map[string]string) *appsv1.DaemonSet {
	return &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: dsName,
		},
		Spec: appsv1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: label,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: label,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "app",
							Image: image,
							Ports: []v1.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	}
}

func listDaemonPods(c clientset.Interface, ns string, label map[string]string) *v1.PodList {
	selector := labels.Set(label).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	podList, err := c.CoreV1().Pods(ns).List(context.TODO(), options)
	framework.ExpectNoError(err)
	gomega.Expect(len(podList.Items)).To(gomega.BeNumerically(">", 0))
	return podList
}

func separateDaemonSetNodeLabels(labels map[string]string) (map[string]string, map[string]string) {
	daemonSetLabels := map[string]string{}
	otherLabels := map[string]string{}
	for k, v := range labels {
		if strings.HasPrefix(k, daemonsetLabelPrefix) {
			daemonSetLabels[k] = v
		} else {
			otherLabels[k] = v
		}
	}
	return daemonSetLabels, otherLabels
}

func clearDaemonSetNodeLabels(c clientset.Interface) error {
	nodeList, err := e2enode.GetReadySchedulableNodes(c)
	if err != nil {
		return err
	}
	for _, node := range nodeList.Items {
		_, err := setDaemonSetNodeLabels(c, node.Name, map[string]string{})
		if err != nil {
			return err
		}
	}
	return nil
}

// updateNamespaceAnnotations sets node selectors related annotations on tests namespaces to empty
func updateNamespaceAnnotations(c clientset.Interface, nsName string) (*v1.Namespace, error) {
	nsClient := c.CoreV1().Namespaces()

	ns, err := nsClient.Get(context.TODO(), nsName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	if ns.Annotations == nil {
		ns.Annotations = make(map[string]string)
	}

	for _, n := range NamespaceNodeSelectors {
		ns.Annotations[n] = ""
	}

	return nsClient.Update(context.TODO(), ns, metav1.UpdateOptions{})
}

func setDaemonSetNodeLabels(c clientset.Interface, nodeName string, labels map[string]string) (*v1.Node, error) {
	nodeClient := c.CoreV1().Nodes()
	var newNode *v1.Node
	var newLabels map[string]string
	err := wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, func() (bool, error) {
		node, err := nodeClient.Get(context.TODO(), nodeName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		// remove all labels this test is creating
		daemonSetLabels, otherLabels := separateDaemonSetNodeLabels(node.Labels)
		if reflect.DeepEqual(daemonSetLabels, labels) {
			newNode = node
			return true, nil
		}
		node.Labels = otherLabels
		for k, v := range labels {
			node.Labels[k] = v
		}
		newNode, err = nodeClient.Update(context.TODO(), node, metav1.UpdateOptions{})
		if err == nil {
			newLabels, _ = separateDaemonSetNodeLabels(newNode.Labels)
			return true, err
		}
		if se, ok := err.(*apierrors.StatusError); ok && se.ErrStatus.Reason == metav1.StatusReasonConflict {
			framework.Logf("failed to update node due to resource version conflict")
			return false, nil
		}
		return false, err
	})
	if err != nil {
		return nil, err
	} else if len(newLabels) != len(labels) {
		return nil, fmt.Errorf("could not set daemon set test labels as expected")
	}

	return newNode, nil
}

func checkDaemonPodOnNodes(f *framework.Framework, ds *appsv1.DaemonSet, nodeNames []string) func() (bool, error) {
	return func() (bool, error) {
		podList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			framework.Logf("could not get the pod list: %v", err)
			return false, nil
		}
		pods := podList.Items

		nodesToPodCount := make(map[string]int)
		for _, pod := range pods {
			if !metav1.IsControlledBy(&pod, ds) {
				continue
			}
			if pod.DeletionTimestamp != nil {
				continue
			}
			if podutil.IsPodAvailable(&pod, ds.Spec.MinReadySeconds, metav1.Now()) {
				nodesToPodCount[pod.Spec.NodeName]++
			}
		}
		framework.Logf("Number of nodes with available pods: %d", len(nodesToPodCount))

		// Ensure that exactly 1 pod is running on all nodes in nodeNames.
		for _, nodeName := range nodeNames {
			if nodesToPodCount[nodeName] != 1 {
				framework.Logf("Node %s is running more than one daemon pod", nodeName)
				return false, nil
			}
		}

		framework.Logf("Number of running nodes: %d, number of available pods: %d", len(nodeNames), len(nodesToPodCount))
		// Ensure that sizes of the lists are the same. We've verified that every element of nodeNames is in
		// nodesToPodCount, so verifying the lengths are equal ensures that there aren't pods running on any
		// other nodes.
		return len(nodesToPodCount) == len(nodeNames), nil
	}
}

func checkRunningOnAllNodes(f *framework.Framework, ds *appsv1.DaemonSet) func() (bool, error) {
	return func() (bool, error) {
		nodeNames := schedulableNodes(f.ClientSet, ds)
		return checkDaemonPodOnNodes(f, ds, nodeNames)()
	}
}

func schedulableNodes(c clientset.Interface, ds *appsv1.DaemonSet) []string {
	nodeList, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err)
	nodeNames := make([]string, 0)
	for _, node := range nodeList.Items {
		if !canScheduleOnNode(node, ds) {
			framework.Logf("DaemonSet pods can't tolerate node %s with taints %+v, skip checking this node", node.Name, node.Spec.Taints)
			continue
		}
		nodeNames = append(nodeNames, node.Name)
	}
	return nodeNames
}

func checkAtLeastOneNewPod(c clientset.Interface, ns string, label map[string]string, newImage string) func() (bool, error) {
	return func() (bool, error) {
		pods := listDaemonPods(c, ns, label)
		for _, pod := range pods.Items {
			if pod.Spec.Containers[0].Image == newImage {
				return true, nil
			}
		}
		return false, nil
	}
}

// canScheduleOnNode checks if a given DaemonSet can schedule pods on the given node
func canScheduleOnNode(node v1.Node, ds *appsv1.DaemonSet) bool {
	newPod := daemon.NewPod(ds, node.Name)
	fitsNodeName, fitsNodeAffinity, fitsTaints := daemon.Predicates(newPod, &node, node.Spec.Taints)
	return fitsNodeName && fitsNodeAffinity && fitsTaints
}

func checkRunningOnNoNodes(f *framework.Framework, ds *appsv1.DaemonSet) func() (bool, error) {
	return checkDaemonPodOnNodes(f, ds, make([]string, 0))
}

func checkDaemonStatus(f *framework.Framework, dsName string) error {
	ds, err := f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).Get(context.TODO(), dsName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("could not get daemon set from v1")
	}
	desired, scheduled, ready := ds.Status.DesiredNumberScheduled, ds.Status.CurrentNumberScheduled, ds.Status.NumberReady
	if desired != scheduled && desired != ready {
		return fmt.Errorf("error in daemon status. DesiredScheduled: %d, CurrentScheduled: %d, Ready: %d", desired, scheduled, ready)
	}
	return nil
}

func checkDaemonPodsImageAndAvailability(c clientset.Interface, ds *appsv1.DaemonSet, image string, maxUnavailable int) func() (bool, error) {
	return func() (bool, error) {
		podList, err := c.CoreV1().Pods(ds.Namespace).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		pods := podList.Items

		unavailablePods := 0
		nodesToUpdatedPodCount := make(map[string]int)
		for _, pod := range pods {
			if !metav1.IsControlledBy(&pod, ds) {
				continue
			}
			podImage := pod.Spec.Containers[0].Image
			if podImage != image {
				framework.Logf("Wrong image for pod: %s. Expected: %s, got: %s.", pod.Name, image, podImage)
			} else {
				nodesToUpdatedPodCount[pod.Spec.NodeName]++
			}
			if !podutil.IsPodAvailable(&pod, ds.Spec.MinReadySeconds, metav1.Now()) {
				framework.Logf("Pod %s is not available", pod.Name)
				unavailablePods++
			}
		}
		if unavailablePods > maxUnavailable {
			return false, fmt.Errorf("number of unavailable pods: %d is greater than maxUnavailable: %d", unavailablePods, maxUnavailable)
		}
		// Make sure every daemon pod on the node has been updated
		nodeNames := schedulableNodes(c, ds)
		for _, node := range nodeNames {
			if nodesToUpdatedPodCount[node] == 0 {
				return false, nil
			}
		}
		return true, nil
	}
}

func checkDaemonSetPodsLabels(podList *v1.PodList, hash string) {
	for _, pod := range podList.Items {
		podHash := pod.Labels[appsv1.DefaultDaemonSetUniqueLabelKey]
		gomega.Expect(len(podHash)).To(gomega.BeNumerically(">", 0))
		if len(hash) > 0 {
			framework.ExpectEqual(podHash, hash)
		}
	}
}

func waitForHistoryCreated(c clientset.Interface, ns string, label map[string]string, numHistory int) {
	listHistoryFn := func() (bool, error) {
		selector := labels.Set(label).AsSelector()
		options := metav1.ListOptions{LabelSelector: selector.String()}
		historyList, err := c.AppsV1().ControllerRevisions(ns).List(context.TODO(), options)
		if err != nil {
			return false, err
		}
		if len(historyList.Items) == numHistory {
			return true, nil
		}
		framework.Logf("%d/%d controllerrevisions created.", len(historyList.Items), numHistory)
		return false, nil
	}
	err := wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, listHistoryFn)
	framework.ExpectNoError(err, "error waiting for controllerrevisions to be created")
}

func listDaemonHistories(c clientset.Interface, ns string, label map[string]string) *appsv1.ControllerRevisionList {
	selector := labels.Set(label).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	historyList, err := c.AppsV1().ControllerRevisions(ns).List(context.TODO(), options)
	framework.ExpectNoError(err)
	gomega.Expect(len(historyList.Items)).To(gomega.BeNumerically(">", 0))
	return historyList
}

func curHistory(historyList *appsv1.ControllerRevisionList, ds *appsv1.DaemonSet) *appsv1.ControllerRevision {
	var curHistory *appsv1.ControllerRevision
	foundCurHistories := 0
	for i := range historyList.Items {
		history := &historyList.Items[i]
		// Every history should have the hash label
		gomega.Expect(len(history.Labels[appsv1.DefaultDaemonSetUniqueLabelKey])).To(gomega.BeNumerically(">", 0))
		match, err := daemon.Match(ds, history)
		framework.ExpectNoError(err)
		if match {
			curHistory = history
			foundCurHistories++
		}
	}
	framework.ExpectEqual(foundCurHistories, 1)
	gomega.Expect(curHistory).NotTo(gomega.BeNil())
	return curHistory
}

func waitFailedDaemonPodDeleted(c clientset.Interface, pod *v1.Pod) func() (bool, error) {
	return func() (bool, error) {
		if _, err := c.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{}); err != nil {
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			return false, fmt.Errorf("failed to get failed daemon pod %q: %v", pod.Name, err)
		}
		return false, nil
	}
}
