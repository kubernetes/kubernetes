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
	"fmt"
	"reflect"
	"strings"
	"time"

	apps "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
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

// This test must be run in serial because it assumes the Daemon Set pods will
// always get scheduled.  If we run other tests in parallel, this may not
// happen.  In the future, running in parallel may work if we have an eviction
// model which lets the DS controller kick out other pods to make room.
// See http://issues.k8s.io/21767 for more details
var _ = SIGDescribe("Daemon set [Serial]", func() {
	var f *framework.Framework

	AfterEach(func() {
		// Clean up
		daemonsets, err := f.ClientSet.Extensions().DaemonSets(f.Namespace.Name).List(metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "unable to dump DaemonSets")
		if daemonsets != nil && len(daemonsets.Items) > 0 {
			for _, ds := range daemonsets.Items {
				By(fmt.Sprintf("Deleting DaemonSet %q with reaper", ds.Name))
				dsReaper, err := kubectl.ReaperFor(extensionsinternal.Kind("DaemonSet"), f.InternalClientset)
				Expect(err).NotTo(HaveOccurred())
				err = dsReaper.Stop(f.Namespace.Name, ds.Name, 0, nil)
				Expect(err).NotTo(HaveOccurred())
				err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, &ds))
				Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to be reaped")
			}
		}
		if daemonsets, err := f.ClientSet.Extensions().DaemonSets(f.Namespace.Name).List(metav1.ListOptions{}); err == nil {
			framework.Logf("daemonset: %s", runtime.EncodeOrDie(api.Codecs.LegacyCodec(api.Registry.EnabledVersions()...), daemonsets))
		} else {
			framework.Logf("unable to dump daemonsets: %v", err)
		}
		if pods, err := f.ClientSet.Core().Pods(f.Namespace.Name).List(metav1.ListOptions{}); err == nil {
			framework.Logf("pods: %s", runtime.EncodeOrDie(api.Codecs.LegacyCodec(api.Registry.EnabledVersions()...), pods))
		} else {
			framework.Logf("unable to dump pods: %v", err)
		}
		err = clearDaemonSetNodeLabels(f.ClientSet)
		Expect(err).NotTo(HaveOccurred())
	})

	f = framework.NewDefaultFramework("daemonsets")

	image := framework.ServeHostnameImage
	dsName := "daemon-set"

	var ns string
	var c clientset.Interface

	BeforeEach(func() {
		ns = f.Namespace.Name

		c = f.ClientSet
		err := clearDaemonSetNodeLabels(c)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should run and stop simple daemon", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		By(fmt.Sprintf("Creating simple DaemonSet %q", dsName))
		ds, err := c.Extensions().DaemonSets(ns).Create(newDaemonSet(dsName, image, label))
		Expect(err).NotTo(HaveOccurred())

		By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to start")
		err = checkDaemonStatus(f, dsName)
		Expect(err).NotTo(HaveOccurred())

		By("Stop a daemon pod, check that the daemon pod is revived.")
		podList := listDaemonPods(c, ns, label)
		pod := podList.Items[0]
		err = c.Core().Pods(ns).Delete(pod.Name, nil)
		Expect(err).NotTo(HaveOccurred())
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to revive")
	})

	It("should run and stop complex daemon", func() {
		complexLabel := map[string]string{daemonsetNameLabel: dsName}
		nodeSelector := map[string]string{daemonsetColorLabel: "blue"}
		framework.Logf("Creating daemon %q with a node selector", dsName)
		ds := newDaemonSet(dsName, image, complexLabel)
		ds.Spec.Template.Spec.NodeSelector = nodeSelector
		ds, err := c.Extensions().DaemonSets(ns).Create(ds)
		Expect(err).NotTo(HaveOccurred())

		By("Initially, daemon pods should not be running on any nodes.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pods to be running on no nodes")

		By("Change node label to blue, check that daemon pod is launched.")
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodeList.Items)).To(BeNumerically(">", 0))
		newNode, err := setDaemonSetNodeLabels(c, nodeList.Items[0].Name, nodeSelector)
		Expect(err).NotTo(HaveOccurred(), "error setting labels on node")
		daemonSetLabels, _ := separateDaemonSetNodeLabels(newNode.Labels)
		Expect(len(daemonSetLabels)).To(Equal(1))
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodOnNodes(f, ds, []string{newNode.Name}))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pods to be running on new nodes")
		err = checkDaemonStatus(f, dsName)
		Expect(err).NotTo(HaveOccurred())

		By("Update the node label to green, and wait for daemons to be unscheduled")
		nodeSelector[daemonsetColorLabel] = "green"
		greenNode, err := setDaemonSetNodeLabels(c, nodeList.Items[0].Name, nodeSelector)
		Expect(err).NotTo(HaveOccurred(), "error removing labels on node")
		Expect(wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, ds))).
			NotTo(HaveOccurred(), "error waiting for daemon pod to not be running on nodes")

		By("Update DaemonSet node selector to green, and change its update strategy to RollingUpdate")
		patch := fmt.Sprintf(`{"spec":{"template":{"spec":{"nodeSelector":{"%s":"%s"}}},"updateStrategy":{"type":"RollingUpdate"}}}`,
			daemonsetColorLabel, greenNode.Labels[daemonsetColorLabel])
		ds, err = c.Extensions().DaemonSets(ns).Patch(dsName, types.StrategicMergePatchType, []byte(patch))
		Expect(err).NotTo(HaveOccurred(), "error patching daemon set")
		daemonSetLabels, _ = separateDaemonSetNodeLabels(greenNode.Labels)
		Expect(len(daemonSetLabels)).To(Equal(1))
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodOnNodes(f, ds, []string{greenNode.Name}))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pods to be running on new nodes")
		err = checkDaemonStatus(f, dsName)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should run and stop complex daemon with node affinity", func() {
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
		ds, err := c.Extensions().DaemonSets(ns).Create(ds)
		Expect(err).NotTo(HaveOccurred())

		By("Initially, daemon pods should not be running on any nodes.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pods to be running on no nodes")

		By("Change node label to blue, check that daemon pod is launched.")
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodeList.Items)).To(BeNumerically(">", 0))
		newNode, err := setDaemonSetNodeLabels(c, nodeList.Items[0].Name, nodeSelector)
		Expect(err).NotTo(HaveOccurred(), "error setting labels on node")
		daemonSetLabels, _ := separateDaemonSetNodeLabels(newNode.Labels)
		Expect(len(daemonSetLabels)).To(Equal(1))
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodOnNodes(f, ds, []string{newNode.Name}))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pods to be running on new nodes")
		err = checkDaemonStatus(f, dsName)
		Expect(err).NotTo(HaveOccurred())

		By("Remove the node label and wait for daemons to be unscheduled")
		_, err = setDaemonSetNodeLabels(c, nodeList.Items[0].Name, map[string]string{})
		Expect(err).NotTo(HaveOccurred(), "error removing labels on node")
		Expect(wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnNoNodes(f, ds))).
			NotTo(HaveOccurred(), "error waiting for daemon pod to not be running on nodes")
	})

	It("should retry creating failed daemon pods", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		By(fmt.Sprintf("Creating a simple DaemonSet %q", dsName))
		ds, err := c.Extensions().DaemonSets(ns).Create(newDaemonSet(dsName, image, label))
		Expect(err).NotTo(HaveOccurred())

		By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to start")
		err = checkDaemonStatus(f, dsName)
		Expect(err).NotTo(HaveOccurred())

		By("Set a daemon pod's phase to 'Failed', check that the daemon pod is revived.")
		podList := listDaemonPods(c, ns, label)
		pod := podList.Items[0]
		pod.ResourceVersion = ""
		pod.Status.Phase = v1.PodFailed
		_, err = c.Core().Pods(ns).UpdateStatus(&pod)
		Expect(err).NotTo(HaveOccurred(), "error failing a daemon pod")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to revive")
	})

	It("Should not update pod when spec was updated and update strategy is OnDelete", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		framework.Logf("Creating simple daemon set %s", dsName)
		ds, err := c.Extensions().DaemonSets(ns).Create(newDaemonSet(dsName, image, label))
		Expect(err).NotTo(HaveOccurred())
		Expect(ds.Spec.TemplateGeneration).To(Equal(int64(1)))

		By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to start")

		By("Make sure all daemon pods have correct template generation 1")
		templateGeneration := "1"
		err = checkDaemonPodsTemplateGeneration(c, ns, label, "1")
		Expect(err).NotTo(HaveOccurred())

		// Check history and labels
		ds, err = c.Extensions().DaemonSets(ns).Get(ds.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		first := curHistory(listDaemonHistories(c, ns, label), ds)
		firstHash := first.Labels[extensions.DefaultDaemonSetUniqueLabelKey]
		Expect(first.Revision).To(Equal(int64(1)))
		checkDaemonSetPodsLabels(listDaemonPods(c, ns, label), firstHash, templateGeneration)

		By("Update daemon pods image.")
		patch := getDaemonSetImagePatch(ds.Spec.Template.Spec.Containers[0].Name, RedisImage)
		ds, err = c.Extensions().DaemonSets(ns).Patch(dsName, types.StrategicMergePatchType, []byte(patch))
		Expect(err).NotTo(HaveOccurred())
		Expect(ds.Spec.TemplateGeneration).To(Equal(int64(2)))

		By("Check that daemon pods images aren't updated.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodsImageAndAvailability(c, ds, image, 0))
		Expect(err).NotTo(HaveOccurred())

		By("Make sure all daemon pods have correct template generation 1")
		err = checkDaemonPodsTemplateGeneration(c, ns, label, templateGeneration)
		Expect(err).NotTo(HaveOccurred())

		By("Check that daemon pods are still running on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to start")

		// Check history and labels
		ds, err = c.Extensions().DaemonSets(ns).Get(ds.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		cur := curHistory(listDaemonHistories(c, ns, label), ds)
		Expect(cur.Revision).To(Equal(int64(2)))
		Expect(cur.Labels[extensions.DefaultDaemonSetUniqueLabelKey]).NotTo(Equal(firstHash))
		checkDaemonSetPodsLabels(listDaemonPods(c, ns, label), firstHash, templateGeneration)
	})

	It("Should update pod when spec was updated and update strategy is RollingUpdate", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		templateGeneration := int64(999)
		framework.Logf("Creating simple daemon set %s with templateGeneration %d", dsName, templateGeneration)
		ds := newDaemonSet(dsName, image, label)
		ds.Spec.TemplateGeneration = templateGeneration
		ds.Spec.UpdateStrategy = extensions.DaemonSetUpdateStrategy{Type: extensions.RollingUpdateDaemonSetStrategyType}
		ds, err := c.Extensions().DaemonSets(ns).Create(ds)
		Expect(err).NotTo(HaveOccurred())
		Expect(ds.Spec.TemplateGeneration).To(Equal(templateGeneration))

		By("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to start")

		By(fmt.Sprintf("Make sure all daemon pods have correct template generation %d", templateGeneration))
		err = checkDaemonPodsTemplateGeneration(c, ns, label, fmt.Sprint(templateGeneration))
		Expect(err).NotTo(HaveOccurred())

		// Check history and labels
		ds, err = c.Extensions().DaemonSets(ns).Get(ds.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		cur := curHistory(listDaemonHistories(c, ns, label), ds)
		hash := cur.Labels[extensions.DefaultDaemonSetUniqueLabelKey]
		Expect(cur.Revision).To(Equal(int64(1)))
		checkDaemonSetPodsLabels(listDaemonPods(c, ns, label), hash, fmt.Sprint(templateGeneration))

		By("Update daemon pods image.")
		patch := getDaemonSetImagePatch(ds.Spec.Template.Spec.Containers[0].Name, RedisImage)
		ds, err = c.Extensions().DaemonSets(ns).Patch(dsName, types.StrategicMergePatchType, []byte(patch))
		Expect(err).NotTo(HaveOccurred())
		templateGeneration++
		Expect(ds.Spec.TemplateGeneration).To(Equal(templateGeneration))

		By("Check that daemon pods images are updated.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodsImageAndAvailability(c, ds, RedisImage, 1))
		Expect(err).NotTo(HaveOccurred())

		By(fmt.Sprintf("Make sure all daemon pods have correct template generation %d", templateGeneration))
		err = checkDaemonPodsTemplateGeneration(c, ns, label, fmt.Sprint(templateGeneration))
		Expect(err).NotTo(HaveOccurred())

		By("Check that daemon pods are still running on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to start")

		// Check history and labels
		ds, err = c.Extensions().DaemonSets(ns).Get(ds.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		cur = curHistory(listDaemonHistories(c, ns, label), ds)
		hash = cur.Labels[extensions.DefaultDaemonSetUniqueLabelKey]
		Expect(cur.Revision).To(Equal(int64(2)))
		checkDaemonSetPodsLabels(listDaemonPods(c, ns, label), hash, fmt.Sprint(templateGeneration))
	})

	It("Should adopt existing pods when creating a RollingUpdate DaemonSet regardless of templateGeneration", func() {
		label := map[string]string{daemonsetNameLabel: dsName}

		// 1. Create a RollingUpdate DaemonSet
		templateGeneration := int64(999)
		framework.Logf("Creating simple RollingUpdate DaemonSet %s with templateGeneration %d", dsName, templateGeneration)
		ds := newDaemonSet(dsName, image, label)
		ds.Spec.TemplateGeneration = templateGeneration
		ds.Spec.UpdateStrategy = extensions.DaemonSetUpdateStrategy{Type: extensions.RollingUpdateDaemonSetStrategyType}
		ds, err := c.Extensions().DaemonSets(ns).Create(ds)
		Expect(err).NotTo(HaveOccurred())
		Expect(ds.Spec.TemplateGeneration).To(Equal(templateGeneration))

		framework.Logf("Check that daemon pods launch on every node of the cluster.")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to start")

		framework.Logf("Make sure all daemon pods have correct template generation %d", templateGeneration)
		err = checkDaemonPodsTemplateGeneration(c, ns, label, fmt.Sprint(templateGeneration))
		Expect(err).NotTo(HaveOccurred())

		// 2. Orphan DaemonSet pods
		framework.Logf("Deleting DaemonSet %s and orphaning its pods and history", dsName)
		deleteDaemonSetAndOrphan(c, ds)

		// 3. Adopt DaemonSet pods (no restart)
		newDSName := "adopt"
		framework.Logf("Creating a new RollingUpdate DaemonSet %s to adopt pods", newDSName)
		newDS := newDaemonSet(newDSName, image, label)
		newDS.Spec.TemplateGeneration = templateGeneration
		newDS.Spec.UpdateStrategy = extensions.DaemonSetUpdateStrategy{Type: extensions.RollingUpdateDaemonSetStrategyType}
		newDS, err = c.Extensions().DaemonSets(ns).Create(newDS)
		Expect(err).NotTo(HaveOccurred())
		Expect(newDS.Spec.TemplateGeneration).To(Equal(templateGeneration))
		Expect(apiequality.Semantic.DeepEqual(newDS.Spec.Template, ds.Spec.Template)).To(BeTrue(), "DaemonSet template should match to adopt pods")

		framework.Logf("Wait for pods and history to be adopted by DaemonSet %s", newDS.Name)
		waitDaemonSetAdoption(c, newDS, ds.Name, templateGeneration)

		// 4. Orphan DaemonSet pods again
		framework.Logf("Deleting DaemonSet %s and orphaning its pods and history", newDSName)
		deleteDaemonSetAndOrphan(c, newDS)

		// 5. Adopt DaemonSet pods (no restart) as long as template matches, even when templateGeneration doesn't match
		newAdoptDSName := "adopt-template-matches"
		framework.Logf("Creating a new RollingUpdate DaemonSet %s to adopt pods", newAdoptDSName)
		newAdoptDS := newDaemonSet(newAdoptDSName, image, label)
		newAdoptDS.Spec.UpdateStrategy = extensions.DaemonSetUpdateStrategy{Type: extensions.RollingUpdateDaemonSetStrategyType}
		newAdoptDS, err = c.Extensions().DaemonSets(ns).Create(newAdoptDS)
		Expect(err).NotTo(HaveOccurred())
		Expect(newAdoptDS.Spec.TemplateGeneration).To(Equal(int64(1)))
		Expect(newAdoptDS.Spec.TemplateGeneration).NotTo(Equal(templateGeneration))
		Expect(apiequality.Semantic.DeepEqual(newAdoptDS.Spec.Template, newDS.Spec.Template)).To(BeTrue(), "DaemonSet template should match to adopt pods")

		framework.Logf(fmt.Sprintf("Wait for pods and history to be adopted by DaemonSet %s", newAdoptDS.Name))
		waitDaemonSetAdoption(c, newAdoptDS, ds.Name, templateGeneration)

		// 6. Orphan DaemonSet pods again
		framework.Logf("Deleting DaemonSet %s and orphaning its pods and history", newAdoptDSName)
		deleteDaemonSetAndOrphan(c, newAdoptDS)

		// 7. Adopt DaemonSet pods (no restart) as long as templateGeneration matches, even when template doesn't match
		newAdoptDSName = "adopt-template-generation-matches"
		framework.Logf("Creating a new RollingUpdate DaemonSet %s to adopt pods", newAdoptDSName)
		newAdoptDS = newDaemonSet(newAdoptDSName, image, label)
		newAdoptDS.Spec.Template.Spec.Containers[0].Name = "not-match"
		newAdoptDS.Spec.UpdateStrategy = extensions.DaemonSetUpdateStrategy{Type: extensions.RollingUpdateDaemonSetStrategyType}
		newAdoptDS.Spec.TemplateGeneration = templateGeneration
		newAdoptDS, err = c.Extensions().DaemonSets(ns).Create(newAdoptDS)
		Expect(err).NotTo(HaveOccurred())
		Expect(newAdoptDS.Spec.TemplateGeneration).To(Equal(templateGeneration))
		Expect(apiequality.Semantic.DeepEqual(newAdoptDS.Spec.Template, newDS.Spec.Template)).NotTo(BeTrue(), "DaemonSet template should not match")

		framework.Logf("Wait for pods and history to be adopted by DaemonSet %s", newAdoptDS.Name)
		waitDaemonSetAdoption(c, newAdoptDS, ds.Name, templateGeneration)
	})

	It("Should rollback without unnecessary restarts", func() {
		// Skip clusters with only one node, where we cannot have half-done DaemonSet rollout for this test
		framework.SkipUnlessNodeCountIsAtLeast(2)

		framework.Logf("Create a RollingUpdate DaemonSet")
		label := map[string]string{daemonsetNameLabel: dsName}
		ds := newDaemonSet(dsName, image, label)
		ds.Spec.UpdateStrategy = extensions.DaemonSetUpdateStrategy{Type: extensions.RollingUpdateDaemonSetStrategyType}
		ds, err := c.Extensions().DaemonSets(ns).Create(ds)
		Expect(err).NotTo(HaveOccurred())

		framework.Logf("Check that daemon pods launch on every node of the cluster")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkRunningOnAllNodes(f, ds))
		Expect(err).NotTo(HaveOccurred(), "error waiting for daemon pod to start")

		framework.Logf("Update the DaemonSet to trigger a rollout")
		// We use a nonexistent image here, so that we make sure it won't finish
		newImage := "foo:non-existent"
		newDS, err := framework.UpdateDaemonSetWithRetries(c, ns, ds.Name, func(update *extensions.DaemonSet) {
			update.Spec.Template.Spec.Containers[0].Image = newImage
		})
		Expect(err).NotTo(HaveOccurred())

		// Make sure we're in the middle of a rollout
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkAtLeastOneNewPod(c, ns, label, newImage))
		Expect(err).NotTo(HaveOccurred())

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
		Expect(len(existingPods)).NotTo(Equal(0))
		Expect(len(newPods)).NotTo(Equal(0))

		framework.Logf("Roll back the DaemonSet before rollout is complete")
		rollbackDS, err := framework.UpdateDaemonSetWithRetries(c, ns, ds.Name, func(update *extensions.DaemonSet) {
			update.Spec.Template.Spec.Containers[0].Image = image
		})
		Expect(err).NotTo(HaveOccurred())

		framework.Logf("Make sure DaemonSet rollback is complete")
		err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonPodsImageAndAvailability(c, rollbackDS, image, 1))
		Expect(err).NotTo(HaveOccurred())

		// After rollback is done, compare current pods with previous old pods during rollout, to make sure they're not restarted
		pods = listDaemonPods(c, ns, label)
		rollbackPods := map[string]bool{}
		for _, pod := range pods.Items {
			rollbackPods[pod.Name] = true
		}
		for _, pod := range existingPods {
			Expect(rollbackPods[pod.Name]).To(BeTrue(), fmt.Sprintf("unexpected pod %s be restarted", pod.Name))
		}
	})
})

// getDaemonSetImagePatch generates a patch for updating a DaemonSet's container image
func getDaemonSetImagePatch(containerName, containerImage string) string {
	return fmt.Sprintf(`{"spec":{"template":{"spec":{"containers":[{"name":"%s","image":"%s"}]}}}}`, containerName, containerImage)
}

// deleteDaemonSetAndOrphan deletes the given DaemonSet and orphans all its dependents.
// It also checks that all dependents are orphaned, and the DaemonSet is deleted.
func deleteDaemonSetAndOrphan(c clientset.Interface, ds *extensions.DaemonSet) {
	trueVar := true
	deleteOptions := &metav1.DeleteOptions{OrphanDependents: &trueVar}
	deleteOptions.Preconditions = metav1.NewUIDPreconditions(string(ds.UID))
	err := c.Extensions().DaemonSets(ds.Namespace).Delete(ds.Name, deleteOptions)
	Expect(err).NotTo(HaveOccurred())

	err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonSetPodsOrphaned(c, ds.Namespace, ds.Spec.Template.Labels))
	Expect(err).NotTo(HaveOccurred(), "error waiting for DaemonSet pods to be orphaned")
	err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonSetHistoryOrphaned(c, ds.Namespace, ds.Spec.Template.Labels))
	Expect(err).NotTo(HaveOccurred(), "error waiting for DaemonSet history to be orphaned")
	err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonSetDeleted(c, ds.Namespace, ds.Name))
	Expect(err).NotTo(HaveOccurred(), "error waiting for DaemonSet to be deleted")
}

func newDaemonSet(dsName, image string, label map[string]string) *extensions.DaemonSet {
	return &extensions.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: dsName,
		},
		Spec: extensions.DaemonSetSpec{
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
	podList, err := c.Core().Pods(ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(podList.Items)).To(BeNumerically(">", 0))
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
	nodeList := framework.GetReadySchedulableNodesOrDie(c)
	for _, node := range nodeList.Items {
		_, err := setDaemonSetNodeLabels(c, node.Name, map[string]string{})
		if err != nil {
			return err
		}
	}
	return nil
}

func setDaemonSetNodeLabels(c clientset.Interface, nodeName string, labels map[string]string) (*v1.Node, error) {
	nodeClient := c.Core().Nodes()
	var newNode *v1.Node
	var newLabels map[string]string
	err := wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, func() (bool, error) {
		node, err := nodeClient.Get(nodeName, metav1.GetOptions{})
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
		newNode, err = nodeClient.Update(node)
		if err == nil {
			newLabels, _ = separateDaemonSetNodeLabels(newNode.Labels)
			return true, err
		}
		if se, ok := err.(*apierrs.StatusError); ok && se.ErrStatus.Reason == metav1.StatusReasonConflict {
			framework.Logf("failed to update node due to resource version conflict")
			return false, nil
		}
		return false, err
	})
	if err != nil {
		return nil, err
	} else if len(newLabels) != len(labels) {
		return nil, fmt.Errorf("Could not set daemon set test labels as expected.")
	}

	return newNode, nil
}

func checkDaemonPodOnNodes(f *framework.Framework, ds *extensions.DaemonSet, nodeNames []string) func() (bool, error) {
	return func() (bool, error) {
		podList, err := f.ClientSet.Core().Pods(f.Namespace.Name).List(metav1.ListOptions{})
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
				nodesToPodCount[pod.Spec.NodeName] += 1
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

func checkRunningOnAllNodes(f *framework.Framework, ds *extensions.DaemonSet) func() (bool, error) {
	return func() (bool, error) {
		nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
		framework.ExpectNoError(err)
		nodeNames := make([]string, 0)
		for _, node := range nodeList.Items {
			if !canScheduleOnNode(node, ds) {
				framework.Logf("DaemonSet pods can't tolerate node %s with taints %+v, skip checking this node", node.Name, node.Spec.Taints)
				continue
			}
			nodeNames = append(nodeNames, node.Name)
		}
		return checkDaemonPodOnNodes(f, ds, nodeNames)()
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

// canScheduleOnNode checks if a given DaemonSet can schedule pods on the given node
func canScheduleOnNode(node v1.Node, ds *extensions.DaemonSet) bool {
	newPod := daemon.NewPod(ds, node.Name)
	nodeInfo := schedulercache.NewNodeInfo()
	nodeInfo.SetNode(&node)
	fit, _, err := daemon.Predicates(newPod, nodeInfo)
	if err != nil {
		framework.Failf("Can't test DaemonSet predicates for node %s: %v", node.Name, err)
		return false
	}
	return fit
}

func checkRunningOnNoNodes(f *framework.Framework, ds *extensions.DaemonSet) func() (bool, error) {
	return checkDaemonPodOnNodes(f, ds, make([]string, 0))
}

func checkDaemonStatus(f *framework.Framework, dsName string) error {
	ds, err := f.ClientSet.Extensions().DaemonSets(f.Namespace.Name).Get(dsName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("Could not get daemon set from v1.")
	}
	desired, scheduled, ready := ds.Status.DesiredNumberScheduled, ds.Status.CurrentNumberScheduled, ds.Status.NumberReady
	if desired != scheduled && desired != ready {
		return fmt.Errorf("Error in daemon status. DesiredScheduled: %d, CurrentScheduled: %d, Ready: %d", desired, scheduled, ready)
	}
	return nil
}

func checkDaemonPodsImageAndAvailability(c clientset.Interface, ds *extensions.DaemonSet, image string, maxUnavailable int) func() (bool, error) {
	return func() (bool, error) {
		podList, err := c.Core().Pods(ds.Namespace).List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		pods := podList.Items

		unavailablePods := 0
		allImagesUpdated := true
		for _, pod := range pods {
			if !metav1.IsControlledBy(&pod, ds) {
				continue
			}
			podImage := pod.Spec.Containers[0].Image
			if podImage != image {
				allImagesUpdated = false
				framework.Logf("Wrong image for pod: %s. Expected: %s, got: %s.", pod.Name, image, podImage)
			}
			if !podutil.IsPodAvailable(&pod, ds.Spec.MinReadySeconds, metav1.Now()) {
				framework.Logf("Pod %s is not available", pod.Name)
				unavailablePods++
			}
		}
		if unavailablePods > maxUnavailable {
			return false, fmt.Errorf("number of unavailable pods: %d is greater than maxUnavailable: %d", unavailablePods, maxUnavailable)
		}
		if !allImagesUpdated {
			return false, nil
		}
		return true, nil
	}
}

func checkDaemonPodsTemplateGeneration(c clientset.Interface, ns string, label map[string]string, templateGeneration string) error {
	pods := listDaemonPods(c, ns, label)
	for _, pod := range pods.Items {
		// We don't care about inactive pods
		if !controller.IsPodActive(&pod) {
			continue
		}
		podTemplateGeneration := pod.Labels[extensions.DaemonSetTemplateGenerationKey]
		if podTemplateGeneration != templateGeneration {
			return fmt.Errorf("expected pod %s/%s template generation %s, but got %s", pod.Namespace, pod.Name, templateGeneration, podTemplateGeneration)
		}
	}
	return nil
}

func checkDaemonSetDeleted(c clientset.Interface, ns, name string) func() (bool, error) {
	return func() (bool, error) {
		_, err := c.Extensions().DaemonSets(ns).Get(name, metav1.GetOptions{})
		if !apierrs.IsNotFound(err) {
			return false, err
		}
		return true, nil
	}
}

func checkDaemonSetPodsOrphaned(c clientset.Interface, ns string, label map[string]string) func() (bool, error) {
	return func() (bool, error) {
		pods := listDaemonPods(c, ns, label)
		for _, pod := range pods.Items {
			// This pod is orphaned only when controller ref is cleared
			if controllerRef := metav1.GetControllerOf(&pod); controllerRef != nil {
				return false, nil
			}
		}
		return true, nil
	}
}

func checkDaemonSetHistoryOrphaned(c clientset.Interface, ns string, label map[string]string) func() (bool, error) {
	return func() (bool, error) {
		histories := listDaemonHistories(c, ns, label)
		for _, history := range histories.Items {
			// This history is orphaned only when controller ref is cleared
			if controllerRef := metav1.GetControllerOf(&history); controllerRef != nil {
				return false, nil
			}
		}
		return true, nil
	}
}

func checkDaemonSetPodsAdopted(c clientset.Interface, ns string, dsUID types.UID, label map[string]string) func() (bool, error) {
	return func() (bool, error) {
		pods := listDaemonPods(c, ns, label)
		for _, pod := range pods.Items {
			// This pod is adopted only when its controller ref is update
			if controllerRef := metav1.GetControllerOf(&pod); controllerRef == nil || controllerRef.UID != dsUID {
				return false, nil
			}
		}
		return true, nil
	}
}

func checkDaemonSetHistoryAdopted(c clientset.Interface, ns string, dsUID types.UID, label map[string]string) func() (bool, error) {
	return func() (bool, error) {
		histories := listDaemonHistories(c, ns, label)
		for _, history := range histories.Items {
			// This history is adopted only when its controller ref is update
			if controllerRef := metav1.GetControllerOf(&history); controllerRef == nil || controllerRef.UID != dsUID {
				return false, nil
			}
		}
		return true, nil
	}
}

func waitDaemonSetAdoption(c clientset.Interface, ds *extensions.DaemonSet, podPrefix string, podTemplateGeneration int64) {
	ns := ds.Namespace
	label := ds.Spec.Template.Labels

	err := wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonSetPodsAdopted(c, ns, ds.UID, label))
	Expect(err).NotTo(HaveOccurred(), "error waiting for DaemonSet pods to be adopted")
	err = wait.PollImmediate(dsRetryPeriod, dsRetryTimeout, checkDaemonSetHistoryAdopted(c, ns, ds.UID, label))
	Expect(err).NotTo(HaveOccurred(), "error waiting for DaemonSet history to be adopted")

	framework.Logf("Make sure no daemon pod updated its template generation %d", podTemplateGeneration)
	err = checkDaemonPodsTemplateGeneration(c, ns, label, fmt.Sprint(podTemplateGeneration))
	Expect(err).NotTo(HaveOccurred())

	framework.Logf("Make sure no pods are recreated by looking at their names")
	err = checkDaemonSetPodsName(c, ns, podPrefix, label)
	Expect(err).NotTo(HaveOccurred())
}

func checkDaemonSetPodsName(c clientset.Interface, ns, prefix string, label map[string]string) error {
	pods := listDaemonPods(c, ns, label)
	for _, pod := range pods.Items {
		if !strings.HasPrefix(pod.Name, prefix) {
			return fmt.Errorf("expected pod %s name to be prefixed %q", pod.Name, prefix)
		}
	}
	return nil
}

func checkDaemonSetPodsLabels(podList *v1.PodList, hash, templateGeneration string) {
	for _, pod := range podList.Items {
		podHash := pod.Labels[extensions.DefaultDaemonSetUniqueLabelKey]
		podTemplate := pod.Labels[extensions.DaemonSetTemplateGenerationKey]
		Expect(len(podHash)).To(BeNumerically(">", 0))
		if len(hash) > 0 {
			Expect(podHash).To(Equal(hash))
		}
		Expect(len(podTemplate)).To(BeNumerically(">", 0))
		Expect(podTemplate).To(Equal(templateGeneration))
	}
}

func listDaemonHistories(c clientset.Interface, ns string, label map[string]string) *apps.ControllerRevisionList {
	selector := labels.Set(label).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	historyList, err := c.AppsV1beta1().ControllerRevisions(ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(historyList.Items)).To(BeNumerically(">", 0))
	return historyList
}

func curHistory(historyList *apps.ControllerRevisionList, ds *extensions.DaemonSet) *apps.ControllerRevision {
	var curHistory *apps.ControllerRevision
	foundCurHistories := 0
	for i := range historyList.Items {
		history := &historyList.Items[i]
		// Every history should have the hash label
		Expect(len(history.Labels[extensions.DefaultDaemonSetUniqueLabelKey])).To(BeNumerically(">", 0))
		match, err := daemon.Match(ds, history)
		Expect(err).NotTo(HaveOccurred())
		if match {
			curHistory = history
			foundCurHistories++
		}
	}
	Expect(foundCurHistories).To(Equal(1))
	Expect(curHistory).NotTo(BeNil())
	return curHistory
}
