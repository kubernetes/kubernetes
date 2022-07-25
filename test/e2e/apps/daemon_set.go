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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"k8s.io/client-go/tools/cache"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	watch "k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/client-go/util/retry"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edaemonset "k8s.io/kubernetes/test/e2e/framework/daemonset"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	admissionapi "k8s.io/pod-security-admission/api"
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
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	image := WebserverImage
	dsName := "daemon-set"

	var ns string
	var c clientset.Interface

	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name

		c = f.ClientSet

		updatedNS, err := patchNamespaceAnnotations(c, ns)
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
		err = e2edaemonset.CheckDaemonStatus(f, dsName)
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
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, e2edaemonset.CheckDaemonPodOnNodes(f, ds, []string{newNode.Name}))
		framework.ExpectNoError(err, "error waiting for daemon pods to be running on new nodes")
		err = e2edaemonset.CheckDaemonStatus(f, dsName)
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
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, e2edaemonset.CheckDaemonPodOnNodes(f, ds, []string{greenNode.Name}))
		framework.ExpectNoError(err, "error waiting for daemon pods to be running on new nodes")
		err = e2edaemonset.CheckDaemonStatus(f, dsName)
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
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, e2edaemonset.CheckDaemonPodOnNodes(f, ds, []string{newNode.Name}))
		framework.ExpectNoError(err, "error waiting for daemon pods to be running on new nodes")
		err = e2edaemonset.CheckDaemonStatus(f, dsName)
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
		err = e2edaemonset.CheckDaemonStatus(f, dsName)
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

	// TODO: This test is expected to be promoted to conformance after the feature is promoted
	ginkgo.It("should surge pods onto nodes when spec was updated and update strategy is RollingUpdate", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		framework.Logf("Creating surge daemon set %s", dsName)
		maxSurgeOverlap := 60 * time.Second
		maxSurge := 1
		surgePercent := intstr.FromString("20%")
		zero := intstr.FromInt(0)
		oldVersion := "1"
		ds := newDaemonSet(dsName, image, label)
		ds.Spec.Template.Spec.Containers[0].Env = []v1.EnvVar{
			{Name: "VERSION", Value: oldVersion},
		}
		// delay shutdown by 15s to allow containers to overlap in time
		ds.Spec.Template.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
			PreStop: &v1.LifecycleHandler{
				Exec: &v1.ExecAction{
					Command: []string{"/bin/sh", "-c", "sleep 15"},
				},
			},
		}
		// use a readiness probe that can be forced to fail (by changing the contents of /var/tmp/ready)
		ds.Spec.Template.Spec.Containers[0].ReadinessProbe = &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				Exec: &v1.ExecAction{
					Command: []string{"/bin/sh", "-ec", `touch /var/tmp/ready; [[ "$( cat /var/tmp/ready )" == "" ]]`},
				},
			},
			InitialDelaySeconds: 7,
			PeriodSeconds:       3,
			SuccessThreshold:    1,
			FailureThreshold:    1,
		}
		// use a simple surge strategy
		ds.Spec.UpdateStrategy = appsv1.DaemonSetUpdateStrategy{
			Type: appsv1.RollingUpdateDaemonSetStrategyType,
			RollingUpdate: &appsv1.RollingUpdateDaemonSet{
				MaxUnavailable: &zero,
				MaxSurge:       &surgePercent,
			},
		}
		// The pod must be ready for at least 10s before we delete the old pod
		ds.Spec.MinReadySeconds = 10

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

		newVersion := "2"
		ginkgo.By("Update daemon pods environment var")
		patch := fmt.Sprintf(`{"spec":{"template":{"spec":{"containers":[{"name":"%s","env":[{"name":"VERSION","value":"%s"}]}]}}}}`, ds.Spec.Template.Spec.Containers[0].Name, newVersion)
		ds, err = c.AppsV1().DaemonSets(ns).Patch(context.TODO(), dsName, types.StrategicMergePatchType, []byte(patch), metav1.PatchOptions{})
		framework.ExpectNoError(err)

		// Time to complete the rolling upgrade is proportional to the number of nodes in the cluster.
		// Get the number of nodes, and set the timeout appropriately.
		nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err)
		nodeCount := len(nodes.Items)
		retryTimeout := dsRetryTimeout + time.Duration(nodeCount*30)*time.Second

		ginkgo.By("Check that daemon pods surge and invariants are preserved during that rollout")
		ageOfOldPod := make(map[string]time.Time)
		deliberatelyDeletedPods := sets.NewString()
		err = wait.PollImmediate(dsRetryPeriod, retryTimeout, func() (bool, error) {
			podList, err := c.CoreV1().Pods(ds.Namespace).List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				return false, err
			}
			pods := podList.Items

			var buf bytes.Buffer
			pw := tabwriter.NewWriter(&buf, 1, 1, 1, ' ', 0)
			fmt.Fprint(pw, "Node\tVersion\tName\tUID\tDeleted\tReady\n")

			now := time.Now()
			podUIDs := sets.NewString()
			deletedPodUIDs := sets.NewString()
			nodes := sets.NewString()
			versions := sets.NewString()
			nodesToVersions := make(map[string]map[string]int)
			nodesToDeletedVersions := make(map[string]map[string]int)
			var surgeCount, newUnavailableCount, newDeliberatelyDeletedCount, oldUnavailableCount, nodesWithoutOldVersion int
			for _, pod := range pods {
				if !metav1.IsControlledBy(&pod, ds) {
					continue
				}
				nodeName := pod.Spec.NodeName
				nodes.Insert(nodeName)
				podVersion := pod.Spec.Containers[0].Env[0].Value
				if pod.DeletionTimestamp != nil {
					if !deliberatelyDeletedPods.Has(string(pod.UID)) {
						versions := nodesToDeletedVersions[nodeName]
						if versions == nil {
							versions = make(map[string]int)
							nodesToDeletedVersions[nodeName] = versions
						}
						versions[podVersion]++
					}
				} else {
					versions := nodesToVersions[nodeName]
					if versions == nil {
						versions = make(map[string]int)
						nodesToVersions[nodeName] = versions
					}
					versions[podVersion]++
				}

				ready := podutil.IsPodAvailable(&pod, ds.Spec.MinReadySeconds, metav1.Now())
				if podVersion == newVersion {
					surgeCount++
					if !ready || pod.DeletionTimestamp != nil {
						if deliberatelyDeletedPods.Has(string(pod.UID)) {
							newDeliberatelyDeletedCount++
						}
						newUnavailableCount++
					}
				} else {
					if !ready || pod.DeletionTimestamp != nil {
						oldUnavailableCount++
					}
				}
				fmt.Fprintf(pw, "%s\t%s\t%s\t%s\t%t\t%t\n", pod.Spec.NodeName, podVersion, pod.Name, pod.UID, pod.DeletionTimestamp != nil, ready)
			}

			// print a stable sorted list of pods by node for debugging
			pw.Flush()
			lines := strings.Split(buf.String(), "\n")
			lines = lines[:len(lines)-1]
			sort.Strings(lines[1:])
			for _, line := range lines {
				framework.Logf("%s", line)
			}

			// if there is an old and new pod at the same time, record a timestamp
			deletedPerNode := make(map[string]int)
			for _, pod := range pods {
				if !metav1.IsControlledBy(&pod, ds) {
					continue
				}
				// ignore deleted pods
				if pod.DeletionTimestamp != nil {
					deletedPodUIDs.Insert(string(pod.UID))
					if !deliberatelyDeletedPods.Has(string(pod.UID)) {
						deletedPerNode[pod.Spec.NodeName]++
					}
					continue
				}
				podUIDs.Insert(string(pod.UID))
				podVersion := pod.Spec.Containers[0].Env[0].Value
				if podVersion == newVersion {
					continue
				}
				// if this is a pod in an older version AND there is a new version of this pod, record when
				// we started seeing this, otherwise delete the record (perhaps the node was drained)
				if nodesToVersions[pod.Spec.NodeName][newVersion] > 0 {
					if _, ok := ageOfOldPod[string(pod.UID)]; !ok {
						ageOfOldPod[string(pod.UID)] = now
					}
				} else {
					delete(ageOfOldPod, string(pod.UID))
				}
			}
			// purge the old pods list of any deleted pods
			for uid := range ageOfOldPod {
				if !podUIDs.Has(uid) {
					delete(ageOfOldPod, uid)
				}
			}
			deliberatelyDeletedPods = deliberatelyDeletedPods.Intersection(deletedPodUIDs)

			for _, versions := range nodesToVersions {
				if versions[oldVersion] == 0 {
					nodesWithoutOldVersion++
				}
			}

			var errs []string

			// invariant: we should not see more than 1 deleted pod per node unless a severe node problem is occurring or the controller is misbehaving
			for node, count := range deletedPerNode {
				if count > 1 {
					errs = append(errs, fmt.Sprintf("Node %s has %d deleted pods, which may indicate a problem on the node or a controller race condition", node, count))
				}
			}

			// invariant: the controller must react to the new pod becoming ready within a reasonable timeframe (2x grace period)
			for uid, firstSeen := range ageOfOldPod {
				if now.Sub(firstSeen) > maxSurgeOverlap {
					errs = append(errs, fmt.Sprintf("An old pod with UID %s has been running alongside a newer version for longer than %s", uid, maxSurgeOverlap))
				}
			}

			// invariant: we should never have more than maxSurge + oldUnavailableCount instances of the new version unready unless a flake in the infrastructure happens, or
			//            if we deliberately deleted one of the new pods
			if newUnavailableCount > (maxSurge + oldUnavailableCount + newDeliberatelyDeletedCount + nodesWithoutOldVersion) {
				errs = append(errs, fmt.Sprintf("observed %d new unavailable pods greater than (surge count %d + old unavailable count %d + deliberately deleted new count %d + nodes without old version %d), may be infrastructure flake", newUnavailableCount, maxSurge, oldUnavailableCount, newDeliberatelyDeletedCount, nodesWithoutOldVersion))
			}
			// invariant: the total number of versions created should be 2
			if versions.Len() > 2 {
				errs = append(errs, fmt.Sprintf("observed %d versions running simultaneously, must have max 2", versions.Len()))
			}
			for _, node := range nodes.List() {
				// ignore pods that haven't been scheduled yet
				if len(node) == 0 {
					continue
				}
				versionCount := make(map[string]int)
				// invariant: surge should never have more than one instance of a pod per node running
				for version, count := range nodesToVersions[node] {
					if count > 1 {
						errs = append(errs, fmt.Sprintf("node %s has %d instances of version %s running simultaneously, must have max 1", node, count, version))
					}
					versionCount[version] += count
				}
				// invariant: when surging, the most number of pods we should allow to be deleted is 2 (if we are getting evicted)
				for version, count := range nodesToDeletedVersions[node] {
					if count > 2 {
						errs = append(errs, fmt.Sprintf("node %s has %d deleted instances of version %s running simultaneously, must have max 1", node, count, version))
					}
					versionCount[version] += count
				}
				// invariant: on any node, we should never have more than two instances of a version (if we are getting evicted)
				for version, count := range versionCount {
					if count > 2 {
						errs = append(errs, fmt.Sprintf("node %s has %d total instances of version %s running simultaneously, must have max 2 (one deleted and one running)", node, count, version))
					}
				}
			}

			if len(errs) > 0 {
				sort.Strings(errs)
				return false, fmt.Errorf("invariants were violated during daemonset update:\n%s", strings.Join(errs, "\n"))
			}

			// Make sure every daemon pod on the node has been updated
			nodeNames := e2edaemonset.SchedulableNodes(c, ds)
			for _, node := range nodeNames {
				switch {
				case
					// if we don't have the new version yet
					nodesToVersions[node][newVersion] == 0,
					// if there are more than one version on a node
					len(nodesToVersions[node]) > 1,
					// if there are still any deleted pods
					len(nodesToDeletedVersions[node]) > 0,
					// if any of the new pods are unavailable
					newUnavailableCount > 0:

					// inject a failure randomly to ensure the controller recovers
					switch rand.Intn(25) {
					// cause a random old pod to go unready
					case 0:
						// select a not-deleted pod of the old version
						if pod := randomPod(pods, func(pod *v1.Pod) bool {
							return pod.DeletionTimestamp == nil && oldVersion == pod.Spec.Containers[0].Env[0].Value
						}); pod != nil {
							// make the /tmp/ready file read only, which will cause readiness to fail
							if _, err := framework.RunKubectl(pod.Namespace, "exec", "-c", pod.Spec.Containers[0].Name, pod.Name, "--", "/bin/sh", "-ec", "echo 0 > /var/tmp/ready"); err != nil {
								framework.Logf("Failed to mark pod %s as unready via exec: %v", pod.Name, err)
							} else {
								framework.Logf("Marked old pod %s as unready", pod.Name)
							}
						}
					case 1:
						// delete a random pod
						if pod := randomPod(pods, func(pod *v1.Pod) bool {
							return pod.DeletionTimestamp == nil
						}); pod != nil {
							if err := c.CoreV1().Pods(ds.Namespace).Delete(context.TODO(), pod.Name, metav1.DeleteOptions{}); err != nil {
								framework.Logf("Failed to delete pod %s early: %v", pod.Name, err)
							} else {
								framework.Logf("Deleted pod %s prematurely", pod.Name)
								deliberatelyDeletedPods.Insert(string(pod.UID))
							}
						}
					}

					// then wait
					return false, nil
				}
			}
			return true, nil
		})
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
		Release: v1.22
		Testname: DaemonSet, list and delete a collection of DaemonSets
		Description: When a DaemonSet is created it MUST succeed. It
		MUST succeed when listing DaemonSets via a label selector. It
		MUST succeed when deleting the DaemonSet via deleteCollection.
	*/
	framework.ConformanceIt("should list and delete a collection of DaemonSets", func() {
		label := map[string]string{daemonsetNameLabel: dsName}
		labelSelector := labels.SelectorFromSet(label).String()

		dsClient := f.ClientSet.AppsV1().DaemonSets(ns)
		cs := f.ClientSet
		one := int64(1)

		ginkgo.By(fmt.Sprintf("Creating simple DaemonSet %q", dsName))
		testDaemonset, err := c.AppsV1().DaemonSets(ns).Create(context.TODO(), newDaemonSetWithLabel(dsName, image, label), metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, testDaemonset))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")
		err = e2edaemonset.CheckDaemonStatus(f, dsName)
		framework.ExpectNoError(err)

		ginkgo.By("listing all DeamonSets")
		dsList, err := cs.AppsV1().DaemonSets("").List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector})
		framework.ExpectNoError(err, "failed to list Daemon Sets")
		framework.ExpectEqual(len(dsList.Items), 1, "filtered list wasn't found")

		ginkgo.By("DeleteCollection of the DaemonSets")
		err = dsClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{GracePeriodSeconds: &one}, metav1.ListOptions{LabelSelector: labelSelector})
		framework.ExpectNoError(err, "failed to delete DaemonSets")

		ginkgo.By("Verify that ReplicaSets have been deleted")
		dsList, err = c.AppsV1().DaemonSets("").List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector})
		framework.ExpectNoError(err, "failed to list DaemonSets")
		framework.ExpectEqual(len(dsList.Items), 0, "filtered list should have no daemonset")
	})

	/*	Release: v1.22
		Testname: DaemonSet, status sub-resource
		Description: When a DaemonSet is created it MUST succeed.
		Attempt to read, update and patch its status sub-resource; all
		mutating sub-resource operations MUST be visible to subsequent reads.
	*/
	framework.ConformanceIt("should verify changes to a daemon set status", func() {
		label := map[string]string{daemonsetNameLabel: dsName}
		labelSelector := labels.SelectorFromSet(label).String()

		dsClient := f.ClientSet.AppsV1().DaemonSets(ns)
		cs := f.ClientSet

		w := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = labelSelector
				return dsClient.Watch(context.TODO(), options)
			},
		}

		dsList, err := cs.AppsV1().DaemonSets("").List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector})
		framework.ExpectNoError(err, "failed to list Daemon Sets")

		ginkgo.By(fmt.Sprintf("Creating simple DaemonSet %q", dsName))
		testDaemonset, err := c.AppsV1().DaemonSets(ns).Create(context.TODO(), newDaemonSetWithLabel(dsName, image, label), metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, testDaemonset))
		framework.ExpectNoError(err, "error waiting for daemon pod to start")
		err = e2edaemonset.CheckDaemonStatus(f, dsName)
		framework.ExpectNoError(err)

		ginkgo.By("Getting /status")
		dsResource := schema.GroupVersionResource{Group: "apps", Version: "v1", Resource: "daemonsets"}
		dsStatusUnstructured, err := f.DynamicClient.Resource(dsResource).Namespace(ns).Get(context.TODO(), dsName, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err, "Failed to fetch the status of daemon set %s in namespace %s", dsName, ns)
		dsStatusBytes, err := json.Marshal(dsStatusUnstructured)
		framework.ExpectNoError(err, "Failed to marshal unstructured response. %v", err)

		var dsStatus appsv1.DaemonSet
		err = json.Unmarshal(dsStatusBytes, &dsStatus)
		framework.ExpectNoError(err, "Failed to unmarshal JSON bytes to a daemon set object type")
		framework.Logf("Daemon Set %s has Conditions: %v", dsName, dsStatus.Status.Conditions)

		ginkgo.By("updating the DaemonSet Status")
		var statusToUpdate, updatedStatus *appsv1.DaemonSet

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			statusToUpdate, err = dsClient.Get(context.TODO(), dsName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to retrieve daemon set %s", dsName)

			statusToUpdate.Status.Conditions = append(statusToUpdate.Status.Conditions, appsv1.DaemonSetCondition{
				Type:    "StatusUpdate",
				Status:  "True",
				Reason:  "E2E",
				Message: "Set from e2e test",
			})

			updatedStatus, err = dsClient.UpdateStatus(context.TODO(), statusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "Failed to update status. %v", err)
		framework.Logf("updatedStatus.Conditions: %#v", updatedStatus.Status.Conditions)

		ginkgo.By("watching for the daemon set status to be updated")
		ctx, cancel := context.WithTimeout(context.Background(), dsRetryTimeout)
		defer cancel()
		_, err = watchtools.Until(ctx, dsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if ds, ok := event.Object.(*appsv1.DaemonSet); ok {
				found := ds.ObjectMeta.Name == testDaemonset.ObjectMeta.Name &&
					ds.ObjectMeta.Namespace == testDaemonset.ObjectMeta.Namespace &&
					ds.Labels[daemonsetNameLabel] == dsName
				if !found {
					framework.Logf("Observed daemon set %v in namespace %v with annotations: %v & Conditions: %v", ds.ObjectMeta.Name, ds.ObjectMeta.Namespace, ds.Annotations, ds.Status.Conditions)
					return false, nil
				}
				for _, cond := range ds.Status.Conditions {
					if cond.Type == "StatusUpdate" &&
						cond.Reason == "E2E" &&
						cond.Message == "Set from e2e test" {
						framework.Logf("Found daemon set %v in namespace %v with labels: %v annotations: %v & Conditions: %v", ds.ObjectMeta.Name, ds.ObjectMeta.Namespace, ds.ObjectMeta.Labels, ds.Annotations, ds.Status.Conditions)
						return found, nil
					}
					framework.Logf("Observed daemon set %v in namespace %v with annotations: %v & Conditions: %v", ds.ObjectMeta.Name, ds.ObjectMeta.Namespace, ds.Annotations, ds.Status.Conditions)
				}
			}
			object := strings.Split(fmt.Sprintf("%v", event.Object), "{")[0]
			framework.Logf("Observed %v event: %+v", object, event.Type)
			return false, nil
		})
		framework.ExpectNoError(err, "failed to locate daemon set %v in namespace %v", testDaemonset.ObjectMeta.Name, ns)
		framework.Logf("Daemon set %s has an updated status", dsName)

		ginkgo.By("patching the DaemonSet Status")
		daemonSetStatusPatch := appsv1.DaemonSet{
			Status: appsv1.DaemonSetStatus{
				Conditions: []appsv1.DaemonSetCondition{
					{
						Type:   "StatusPatched",
						Status: "True",
					},
				},
			},
		}

		payload, err := json.Marshal(daemonSetStatusPatch)
		framework.ExpectNoError(err, "Failed to marshal JSON. %v", err)
		_, err = dsClient.Patch(context.TODO(), dsName, types.MergePatchType, payload, metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err, "Failed to patch daemon set status", err)

		ginkgo.By("watching for the daemon set status to be patched")
		ctx, cancel = context.WithTimeout(context.Background(), dsRetryTimeout)
		defer cancel()
		_, err = watchtools.Until(ctx, dsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if ds, ok := event.Object.(*appsv1.DaemonSet); ok {
				found := ds.ObjectMeta.Name == testDaemonset.ObjectMeta.Name &&
					ds.ObjectMeta.Namespace == testDaemonset.ObjectMeta.Namespace &&
					ds.Labels[daemonsetNameLabel] == dsName
				if !found {
					framework.Logf("Observed daemon set %v in namespace %v with annotations: %v & Conditions: %v", ds.ObjectMeta.Name, ds.ObjectMeta.Namespace, ds.Annotations, ds.Status.Conditions)
					return false, nil
				}
				for _, cond := range ds.Status.Conditions {
					if cond.Type == "StatusPatched" {
						framework.Logf("Found daemon set %v in namespace %v with labels: %v annotations: %v & Conditions: %v", ds.ObjectMeta.Name, ds.ObjectMeta.Namespace, ds.ObjectMeta.Labels, ds.Annotations, ds.Status.Conditions)
						return found, nil
					}
					framework.Logf("Observed daemon set %v in namespace %v with annotations: %v & Conditions: %v", ds.ObjectMeta.Name, ds.ObjectMeta.Namespace, ds.Annotations, ds.Status.Conditions)
				}
			}
			object := strings.Split(fmt.Sprintf("%v", event.Object), "{")[0]
			framework.Logf("Observed %v event: %v", object, event.Type)
			return false, nil
		})
		framework.ExpectNoError(err, "failed to locate daemon set %v in namespace %v", testDaemonset.ObjectMeta.Name, ns)
		framework.Logf("Daemon set %s has a patched status", dsName)
	})
})

// randomPod selects a random pod within pods that causes fn to return true, or nil
// if no pod can be found matching the criteria.
func randomPod(pods []v1.Pod, fn func(p *v1.Pod) bool) *v1.Pod {
	podCount := len(pods)
	for offset, i := rand.Intn(podCount), 0; i < (podCount - 1); i++ {
		pod := &pods[(offset+i)%podCount]
		if fn(pod) {
			return pod
		}
	}
	return nil
}

// getDaemonSetImagePatch generates a patch for updating a DaemonSet's container image
func getDaemonSetImagePatch(containerName, containerImage string) string {
	return fmt.Sprintf(`{"spec":{"template":{"spec":{"containers":[{"name":"%s","image":"%s"}]}}}}`, containerName, containerImage)
}

func newDaemonSet(dsName, image string, label map[string]string) *appsv1.DaemonSet {
	ds := newDaemonSetWithLabel(dsName, image, label)
	ds.ObjectMeta.Labels = nil
	return ds
}

func newDaemonSetWithLabel(dsName, image string, label map[string]string) *appsv1.DaemonSet {
	return e2edaemonset.NewDaemonSet(dsName, image, label, nil, nil, []v1.ContainerPort{{ContainerPort: 9376}})
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

// patchNamespaceAnnotations sets node selectors related annotations on tests namespaces to empty
func patchNamespaceAnnotations(c clientset.Interface, nsName string) (*v1.Namespace, error) {
	nsClient := c.CoreV1().Namespaces()

	annotations := make(map[string]string)
	for _, n := range NamespaceNodeSelectors {
		annotations[n] = ""
	}
	nsPatch, err := json.Marshal(map[string]interface{}{
		"metadata": map[string]interface{}{
			"annotations": annotations,
		},
	})
	if err != nil {
		return nil, err
	}

	return nsClient.Patch(context.TODO(), nsName, types.StrategicMergePatchType, nsPatch, metav1.PatchOptions{})
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

func checkRunningOnAllNodes(f *framework.Framework, ds *appsv1.DaemonSet) func() (bool, error) {
	return func() (bool, error) {
		return e2edaemonset.CheckRunningOnAllNodes(f, ds)
	}
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

func checkRunningOnNoNodes(f *framework.Framework, ds *appsv1.DaemonSet) func() (bool, error) {
	return e2edaemonset.CheckDaemonPodOnNodes(f, ds, make([]string, 0))
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
			// Ignore the pod on the node that is supposed to be deleted
			if pod.DeletionTimestamp != nil {
				continue
			}
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
		nodeNames := e2edaemonset.SchedulableNodes(c, ds)
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
		// Ignore all the DS pods that will be deleted
		if pod.DeletionTimestamp != nil {
			continue
		}
		podHash := pod.Labels[appsv1.DefaultDaemonSetUniqueLabelKey]
		gomega.Expect(len(podHash)).To(gomega.BeNumerically(">", 0))
		if len(hash) > 0 {
			framework.ExpectEqual(podHash, hash, "unexpected hash for pod %s", pod.Name)
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
