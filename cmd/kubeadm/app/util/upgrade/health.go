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

	"github.com/pkg/errors"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
)

// healthCheck is a helper struct for easily performing healthchecks against the cluster and printing the output
type healthCheck struct {
	name   string
	client clientset.Interface
	// f is invoked with a k8s client passed to it. Should return an optional error
	f func(clientset.Interface) error
}

// Check is part of the preflight.Checker interface
func (c *healthCheck) Check() (warnings, errors []error) {
	if err := c.f(c.client); err != nil {
		return nil, []error{err}
	}
	return nil, nil
}

// Name is part of the preflight.Checker interface
func (c *healthCheck) Name() string {
	return c.name
}

// CheckClusterHealth makes sure:
// - the API /healthz endpoint is healthy
// - all master Nodes are Ready
// - (if self-hosted) that there are DaemonSets with at least one Pod for all control plane components
// - (if static pod-hosted) that all required Static Pod manifests exist on disk
func CheckClusterHealth(client clientset.Interface, ignoreChecksErrors sets.String) error {
	fmt.Println("[upgrade] Making sure the cluster is healthy:")

	healthChecks := []preflight.Checker{
		&healthCheck{
			name:   "APIServerHealth",
			client: client,
			f:      apiServerHealthy,
		},
		&healthCheck{
			name:   "MasterNodesReady",
			client: client,
			f:      masterNodesReady,
		},
		// TODO: Add a check for ComponentStatuses here?
	}

	healthChecks = append(healthChecks, &healthCheck{
		name:   "StaticPodManifest",
		client: client,
		f:      staticPodManifestHealth,
	})

	return preflight.RunChecks(healthChecks, os.Stderr, ignoreChecksErrors)
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
		return errors.Errorf("the API Server is unhealthy; /healthz didn't return %q", "ok")
	}
	return nil
}

// masterNodesReady checks whether all master Nodes in the cluster are in the Running state
func masterNodesReady(client clientset.Interface) error {
	selector := labels.SelectorFromSet(labels.Set(map[string]string{
		constants.LabelNodeRoleMaster: "",
	}))
	masters, err := client.CoreV1().Nodes().List(metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	if err != nil {
		return errors.Wrap(err, "couldn't list masters in cluster")
	}

	if len(masters.Items) == 0 {
		return errors.New("failed to find any nodes with master role")
	}

	notReadyMasters := getNotReadyNodes(masters.Items)
	if len(notReadyMasters) != 0 {
		return errors.Errorf("there are NotReady masters in the cluster: %v", notReadyMasters)
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
	return errors.Errorf("The control plane seems to be Static Pod-hosted, but some of the manifests don't seem to exist on disk. This probably means you're running 'kubeadm upgrade' on a remote machine, which is not supported for a Static Pod-hosted cluster. Manifest files not found: %v", nonExistentManifests)
}

// IsControlPlaneSelfHosted returns whether the control plane is self hosted or not
func IsControlPlaneSelfHosted(client clientset.Interface) bool {
	notReadyDaemonSets, err := getNotReadyDaemonSets(client)
	if err != nil {
		return false
	}

	// If there are no NotReady DaemonSets, we are using selfhosting
	return len(notReadyDaemonSets) == 0
}

// getNotReadyDaemonSets gets the amount of Ready control plane DaemonSets
func getNotReadyDaemonSets(client clientset.Interface) ([]error, error) {
	notReadyDaemonSets := []error{}
	for _, component := range constants.MasterComponents {
		dsName := constants.AddSelfHostedPrefix(component)
		ds, err := client.AppsV1().DaemonSets(metav1.NamespaceSystem).Get(dsName, metav1.GetOptions{})
		if err != nil {
			return nil, errors.Errorf("couldn't get daemonset %q in the %s namespace", dsName, metav1.NamespaceSystem)
		}

		if err := daemonSetHealth(&ds.Status); err != nil {
			notReadyDaemonSets = append(notReadyDaemonSets, errors.Wrapf(err, "DaemonSet %q not healthy", dsName))
		}
	}
	return notReadyDaemonSets, nil
}

// daemonSetHealth is a helper function for getting the health of a DaemonSet's status
func daemonSetHealth(dsStatus *apps.DaemonSetStatus) error {
	if dsStatus.CurrentNumberScheduled != dsStatus.DesiredNumberScheduled {
		return errors.Errorf("current number of scheduled Pods ('%d') doesn't match the amount of desired Pods ('%d')",
			dsStatus.CurrentNumberScheduled, dsStatus.DesiredNumberScheduled)
	}
	if dsStatus.NumberAvailable == 0 {
		return errors.New("no available Pods for DaemonSet")
	}
	if dsStatus.NumberReady == 0 {
		return errors.New("no ready Pods for DaemonSet")
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
