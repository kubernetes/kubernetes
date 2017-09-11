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
	"net/http"
	"os"

	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// healthCheck is a helper struct for easily performing healthchecks against the cluster and printing the output
type healthCheck struct {
	description, okMessage, failMessage string
	// f is invoked with a k8s client passed to it. Should return an optional warning and/or an error
	f func(clientset.Interface) error
}

// CheckClusterHealth makes sure:
// - the API /healthz endpoint is healthy
// - all Nodes are Ready
// - (if self-hosted) that there are DaemonSets with at least one Pod for all control plane components
// - (if static pod-hosted) that all required Static Pod manifests exist on disk
func CheckClusterHealth(client clientset.Interface) error {
	fmt.Println("[upgrade] Making sure the cluster is healthy:")

	healthChecks := []healthCheck{
		{
			description: "API Server health",
			okMessage:   "Healthy",
			failMessage: "Unhealthy",
			f:           apiServerHealthy,
		},
		{
			description: "Node health",
			okMessage:   "All Nodes are healthy",
			failMessage: "More than one Node unhealthy",
			f:           nodesHealthy,
		},
		// TODO: Add a check for ComponentStatuses here?
	}

	// Run slightly different health checks depending on control plane hosting type
	if IsControlPlaneSelfHosted(client) {
		healthChecks = append(healthChecks, healthCheck{
			description: "Control plane DaemonSet health",
			okMessage:   "All control plane DaemonSets are healthy",
			failMessage: "More than one control plane DaemonSet unhealthy",
			f:           controlPlaneHealth,
		})
	} else {
		healthChecks = append(healthChecks, healthCheck{
			description: "Static Pod manifests exists on disk",
			okMessage:   "All manifests exist on disk",
			failMessage: "Some manifests don't exist on disk",
			f:           staticPodManifestHealth,
		})
	}

	return runHealthChecks(client, healthChecks)
}

// runHealthChecks runs a set of health checks against the cluster
func runHealthChecks(client clientset.Interface, healthChecks []healthCheck) error {
	for _, check := range healthChecks {

		err := check.f(client)
		if err != nil {
			fmt.Printf("[upgrade/health] Checking %s: %s\n", check.description, check.failMessage)
			return fmt.Errorf("The cluster is not in an upgradeable state due to: %v", err)
		}
		fmt.Printf("[upgrade/health] Checking %s: %s\n", check.description, check.okMessage)
	}
	return nil
}

// apiServerHealthy checks whether the API server's /healthz endpoint is healthy
func apiServerHealthy(client clientset.Interface) error {
	healthStatus := 0

	// If client.Discovery().RESTClient() is nil, the fake client is used, and that means we are dry-running. Just proceed
	if client.Discovery().RESTClient() == nil {
		return nil
	}
	client.Discovery().RESTClient().Get().AbsPath("/healthz").Do().StatusCode(&healthStatus)
	if healthStatus != http.StatusOK {
		return fmt.Errorf("the API Server is unhealthy; /healthz didn't return %q", "ok")
	}
	return nil
}

// nodesHealthy checks whether all Nodes in the cluster are in the Running state
func nodesHealthy(client clientset.Interface) error {
	nodes, err := client.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("couldn't list all nodes in cluster: %v", err)
	}

	notReadyNodes := getNotReadyNodes(nodes.Items)
	if len(notReadyNodes) != 0 {
		return fmt.Errorf("there are NotReady Nodes in the cluster: %v", notReadyNodes)
	}
	return nil
}

// controlPlaneHealth ensures all control plane DaemonSets are healthy
func controlPlaneHealth(client clientset.Interface) error {
	notReadyDaemonSets, err := getNotReadyDaemonSets(client)
	if err != nil {
		return err
	}

	if len(notReadyDaemonSets) != 0 {
		return fmt.Errorf("there are control plane DaemonSets in the cluster that are not ready: %v", notReadyDaemonSets)
	}
	return nil
}

// staticPodManifestHealth makes sure the required static pods are presents
func staticPodManifestHealth(_ clientset.Interface) error {
	nonExistentManifests := []string{}
	for _, component := range constants.MasterComponents {
		manifestFile := constants.GetStaticPodFilepath(component, constants.GetStaticPodDirectory())
		if _, err := os.Stat(manifestFile); os.IsNotExist(err) {
			nonExistentManifests = append(nonExistentManifests, manifestFile)
		}
	}
	if len(nonExistentManifests) == 0 {
		return nil
	}
	return fmt.Errorf("The control plane seems to be Static Pod-hosted, but some of the manifests don't seem to exist on disk. This probably means you're running 'kubeadm upgrade' on a remote machine, which is not supported for a Static Pod-hosted cluster. Manifest files not found: %v", nonExistentManifests)
}

// IsControlPlaneSelfHosted returns whether the control plane is self hosted or not
func IsControlPlaneSelfHosted(client clientset.Interface) bool {
	notReadyDaemonSets, err := getNotReadyDaemonSets(client)
	if err != nil {
		return false
	}

	// If there are no NotReady DaemonSets, we are using self-hosting
	return len(notReadyDaemonSets) == 0
}

// getNotReadyDaemonSets gets the amount of Ready control plane DaemonSets
func getNotReadyDaemonSets(client clientset.Interface) ([]error, error) {
	notReadyDaemonSets := []error{}
	for _, component := range constants.MasterComponents {
		dsName := constants.AddSelfHostedPrefix(component)
		ds, err := client.ExtensionsV1beta1().DaemonSets(metav1.NamespaceSystem).Get(dsName, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("couldn't get daemonset %q in the %s namespace", dsName, metav1.NamespaceSystem)
		}

		if err := daemonSetHealth(&ds.Status); err != nil {
			notReadyDaemonSets = append(notReadyDaemonSets, fmt.Errorf("DaemonSet %q not healthy: %v", dsName, err))
		}
	}
	return notReadyDaemonSets, nil
}

// daemonSetHealth is a helper function for getting the health of a DaemonSet's status
func daemonSetHealth(dsStatus *extensions.DaemonSetStatus) error {
	if dsStatus.CurrentNumberScheduled != dsStatus.DesiredNumberScheduled {
		return fmt.Errorf("current number of scheduled Pods ('%d') doesn't match the amount of desired Pods ('%d')", dsStatus.CurrentNumberScheduled, dsStatus.DesiredNumberScheduled)
	}
	if dsStatus.NumberAvailable == 0 {
		return fmt.Errorf("no available Pods for DaemonSet")
	}
	if dsStatus.NumberReady == 0 {
		return fmt.Errorf("no ready Pods for DaemonSet")
	}
	return nil
}

// getNotReadyNodes returns a string slice of nodes in the cluster that are NotReady
func getNotReadyNodes(nodes []v1.Node) []string {
	notReadyNodes := []string{}
	for _, node := range nodes {
		for _, condition := range node.Status.Conditions {
			if condition.Type == v1.NodeReady && condition.Status != v1.ConditionTrue {
				notReadyNodes = append(notReadyNodes, node.ObjectMeta.Name)
			}
		}
	}
	return notReadyNodes
}
