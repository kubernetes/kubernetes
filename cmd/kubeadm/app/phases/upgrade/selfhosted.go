/*
Copyright 2017 The Kubernetes Authors.

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

package upgrade

import (
	"fmt"
	"time"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/selfhosting"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

const (
	// upgradeTempDSPrefix is the prefix added to the temporary DaemonSet's name used during the upgrade
	upgradeTempDSPrefix = "temp-upgrade-"

	// upgradeTempLabel is the label key used for identifying the temporary component's DaemonSet
	upgradeTempLabel = "temp-upgrade-component"

	// selfHostingWaitTimeout describes the maximum amount of time a self-hosting wait process should wait before timing out
	selfHostingWaitTimeout = 2 * time.Minute

	// selfHostingFailureThreshold describes how many times kubeadm will retry creating the DaemonSets
	selfHostingFailureThreshold int = 10
)

// controlPlaneComponentResources holds the relevant Pod and DaemonSet associated with a control plane component
type controlPlaneComponentResources struct {
	pod       *v1.Pod
	daemonSet *apps.DaemonSet
}

// SelfHostedControlPlane upgrades a self-hosted control plane
// It works as follows:
// - The client gets the currently running DaemonSets and their associated Pods used for self-hosting the control plane
// - A temporary DaemonSet for the component in question is created; but nearly identical to the DaemonSet for the self-hosted component running right now
//    - Why use this temporary DaemonSet? Because, the RollingUpdate strategy for upgrading DaemonSets first kills the old Pod, and then adds the new one
//    - This doesn't work for self-hosted upgrades, as if you remove the only API server for instance you have in the cluster, the cluster essentially goes down
//    - So instead, a nearly identical copy of the pre-upgrade DaemonSet is created and applied to the cluster. In the beginning, this duplicate DS is just idle
// - kubeadm waits for the temporary DaemonSet's Pod to become Running
// - kubeadm updates the real, self-hosted component. This will result in the pre-upgrade component Pod being removed from the cluster
//    - Luckily, the temporary, backup DaemonSet now kicks in and takes over and acts as the control plane. It recognizes that a new Pod should be created,
//    - as the "real" DaemonSet is being updated.
// - kubeadm waits for the pre-upgrade Pod to become deleted. It now takes advantage of the backup/temporary component
// - kubeadm waits for the new, upgraded DaemonSet to become Running.
// - Now that the new, upgraded DaemonSet is Running, we can delete the backup/temporary DaemonSet
// - Lastly, make sure the API /healthz endpoint still is reachable
//
// TL;DR; This is what the flow looks like in pseudo-code:
// for [kube-apiserver, kube-controller-manager, kube-scheduler], do:
//    1. Self-Hosted component v1 Running
//       -> Duplicate the DaemonSet manifest
//    2. Self-Hosted component v1 Running (active). Backup component v1 Running (passive)
//       -> Upgrade the Self-Hosted component v1 to v2.
//       -> Self-Hosted component v1 is Deleted from the cluster
//    3. Backup component v1 Running becomes active and completes the upgrade by creating the Self-Hosted component v2 Pod (passive)
//       -> Wait for Self-Hosted component v2 to become Running
//    4. Backup component v1 Running (active). Self-Hosted component v2 Running (passive)
//       -> Backup component v1 is Deleted
//    5. Wait for Self-Hosted component v2 Running to become active
//    6. Repeat for all control plane components
func SelfHostedControlPlane(client clientset.Interface, waiter apiclient.Waiter, cfg *kubeadmapi.InitConfiguration, k8sVersion *version.Version) error {

	// Adjust the timeout slightly to something self-hosting specific
	waiter.SetTimeout(selfHostingWaitTimeout)

	// This function returns a map of DaemonSet objects ready to post to the API server
	newControlPlaneDaemonSets := BuildUpgradedDaemonSetsFromConfig(cfg, k8sVersion)

	controlPlaneResources, err := getCurrentControlPlaneComponentResources(client)
	if err != nil {
		return err
	}

	for _, component := range constants.MasterComponents {
		// Make a shallow copy of the current DaemonSet in order to create a new, temporary one
		tempDS := *controlPlaneResources[component].daemonSet

		// Mutate the temp daemonset a little to be suitable for this usage (change label selectors, etc)
		mutateTempDaemonSet(&tempDS, component)

		// Create or update the DaemonSet in the API Server, and retry selfHostingFailureThreshold times if it errors out
		if err := apiclient.TryRunCommand(func() error {
			return apiclient.CreateOrUpdateDaemonSet(client, &tempDS)
		}, selfHostingFailureThreshold); err != nil {
			return err
		}

		// Wait for the temporary/backup self-hosted component to come up
		if err := waiter.WaitForPodsWithLabel(buildTempUpgradeDSLabelQuery(component)); err != nil {
			return err
		}

		newDS := newControlPlaneDaemonSets[component]

		// Upgrade the component's self-hosted resource
		// During this upgrade; the temporary/backup component will take over
		if err := apiclient.TryRunCommand(func() error {

			if _, err := client.AppsV1().DaemonSets(newDS.ObjectMeta.Namespace).Update(newDS); err != nil {
				return fmt.Errorf("couldn't update self-hosted component's DaemonSet: %v", err)
			}
			return nil
		}, selfHostingFailureThreshold); err != nil {
			return err
		}

		// Wait for the component's old Pod to disappear
		oldPod := controlPlaneResources[component].pod
		if err := waiter.WaitForPodToDisappear(oldPod.ObjectMeta.Name); err != nil {
			return err
		}

		// Wait for the main, upgraded self-hosted component to come up
		// Here we're talking to the temporary/backup component; the upgraded component is in the process of starting up
		if err := waiter.WaitForPodsWithLabel(selfhosting.BuildSelfHostedComponentLabelQuery(component)); err != nil {
			return err
		}

		// Delete the temporary DaemonSet, and retry selfHostingFailureThreshold times if it errors out
		// In order to pivot back to the upgraded API server, we kill the temporary/backup component
		if err := apiclient.TryRunCommand(func() error {
			return apiclient.DeleteDaemonSetForeground(client, tempDS.ObjectMeta.Namespace, tempDS.ObjectMeta.Name)
		}, selfHostingFailureThreshold); err != nil {
			return err
		}

		// Just as an extra safety check; make sure the API server is returning ok at the /healthz endpoint
		if err := waiter.WaitForAPI(); err != nil {
			return err
		}

		fmt.Printf("[upgrade/apply] Self-hosted component %q upgraded successfully!\n", component)
	}
	return nil
}

// BuildUpgradedDaemonSetsFromConfig takes a config object and the current version and returns the DaemonSet objects to post to the master
func BuildUpgradedDaemonSetsFromConfig(cfg *kubeadmapi.InitConfiguration, k8sVersion *version.Version) map[string]*apps.DaemonSet {
	// Here the map of different mutators to use for the control plane's podspec is stored
	mutators := selfhosting.GetMutatorsFromFeatureGates(cfg.FeatureGates)
	// Get the new PodSpecs to use
	controlPlanePods := controlplane.GetStaticPodSpecs(cfg, k8sVersion)
	// Store the created DaemonSets in this map
	controlPlaneDaemonSets := map[string]*apps.DaemonSet{}

	for _, component := range constants.MasterComponents {
		podSpec := controlPlanePods[component].Spec

		// Build the full DaemonSet object from the PodSpec generated from the control plane phase and
		// using the self-hosting mutators available from the selfhosting phase
		ds := selfhosting.BuildDaemonSet(component, &podSpec, mutators)
		controlPlaneDaemonSets[component] = ds
	}
	return controlPlaneDaemonSets
}

// addTempUpgradeDSPrefix adds the upgradeTempDSPrefix to the specified DaemonSet name
func addTempUpgradeDSPrefix(currentName string) string {
	return fmt.Sprintf("%s%s", upgradeTempDSPrefix, currentName)
}

// buildTempUpgradeLabels returns the label string-string map for identifying the temporary
func buildTempUpgradeLabels(component string) map[string]string {
	return map[string]string{
		upgradeTempLabel: component,
	}
}

// buildTempUpgradeDSLabelQuery creates the right query for matching
func buildTempUpgradeDSLabelQuery(component string) string {
	return fmt.Sprintf("%s=%s", upgradeTempLabel, component)
}

// mutateTempDaemonSet mutates the specified self-hosted DaemonSet for the specified component
// in a way that makes it possible to post a nearly identical, temporary DaemonSet as a backup
func mutateTempDaemonSet(tempDS *apps.DaemonSet, component string) {
	// Prefix the name of the temporary DaemonSet with upgradeTempDSPrefix
	tempDS.ObjectMeta.Name = addTempUpgradeDSPrefix(tempDS.ObjectMeta.Name)
	// Set .Labels to something else than the "real" self-hosted components have
	tempDS.ObjectMeta.Labels = buildTempUpgradeLabels(component)
	tempDS.Spec.Selector.MatchLabels = buildTempUpgradeLabels(component)
	tempDS.Spec.Template.ObjectMeta.Labels = buildTempUpgradeLabels(component)
	// Clean all unnecessary ObjectMeta fields
	tempDS.ObjectMeta = extractRelevantObjectMeta(tempDS.ObjectMeta)
	// Reset .Status as we're posting a new object
	tempDS.Status = apps.DaemonSetStatus{}
}

// extractRelevantObjectMeta returns only the relevant parts of ObjectMeta required when creating
// a new, identical resource. We should not POST ResourceVersion, UUIDs, etc., only the name, labels,
// namespace and annotations should be preserved.
func extractRelevantObjectMeta(ob metav1.ObjectMeta) metav1.ObjectMeta {
	return metav1.ObjectMeta{
		Name:        ob.Name,
		Namespace:   ob.Namespace,
		Labels:      ob.Labels,
		Annotations: ob.Annotations,
	}
}

// listPodsWithLabelSelector returns the relevant Pods for the given LabelSelector
func listPodsWithLabelSelector(client clientset.Interface, kvLabel string) (*v1.PodList, error) {
	return client.CoreV1().Pods(metav1.NamespaceSystem).List(metav1.ListOptions{
		LabelSelector: kvLabel,
	})
}

// getCurrentControlPlaneComponentResources returns a string-(Pod|DaemonSet) map for later use
func getCurrentControlPlaneComponentResources(client clientset.Interface) (map[string]controlPlaneComponentResources, error) {
	controlPlaneResources := map[string]controlPlaneComponentResources{}

	for _, component := range constants.MasterComponents {
		var podList *v1.PodList
		var currentDS *apps.DaemonSet

		// Get the self-hosted pod associated with the component
		podLabelSelector := selfhosting.BuildSelfHostedComponentLabelQuery(component)
		if err := apiclient.TryRunCommand(func() error {
			var tryrunerr error
			podList, tryrunerr = listPodsWithLabelSelector(client, podLabelSelector)
			return tryrunerr // note that tryrunerr is most likely nil here (in successful cases)
		}, selfHostingFailureThreshold); err != nil {
			return nil, err
		}

		// Make sure that there are only one Pod with this label selector; otherwise unexpected things can happen
		if len(podList.Items) > 1 {
			return nil, fmt.Errorf("too many pods with label selector %q found in the %s namespace", podLabelSelector, metav1.NamespaceSystem)
		}

		// Get the component's DaemonSet object
		dsName := constants.AddSelfHostedPrefix(component)
		if err := apiclient.TryRunCommand(func() error {
			var tryrunerr error
			// Try to get the current self-hosted component
			currentDS, tryrunerr = client.AppsV1().DaemonSets(metav1.NamespaceSystem).Get(dsName, metav1.GetOptions{})
			return tryrunerr // note that tryrunerr is most likely nil here (in successful cases)
		}, selfHostingFailureThreshold); err != nil {
			return nil, err
		}

		// Add the associated resources to the map to return later
		controlPlaneResources[component] = controlPlaneComponentResources{
			pod:       &podList.Items[0],
			daemonSet: currentDS,
		}
	}
	return controlPlaneResources, nil
}
