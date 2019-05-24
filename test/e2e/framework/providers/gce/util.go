/*
Copyright 2019 The Kubernetes Authors.

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

package gce

import (
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// RecreateNodes recreates the given nodes in a managed instance group.
func RecreateNodes(c clientset.Interface, nodes []v1.Node) error {
	// Build mapping from zone to nodes in that zone.
	nodeNamesByZone := make(map[string][]string)
	for i := range nodes {
		node := &nodes[i]
		zone := framework.TestContext.CloudConfig.Zone
		if z, ok := node.Labels[v1.LabelZoneFailureDomain]; ok {
			zone = z
		}
		nodeNamesByZone[zone] = append(nodeNamesByZone[zone], node.Name)
	}

	// Find the sole managed instance group name
	var instanceGroup string
	if strings.Index(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") >= 0 {
		return fmt.Errorf("Test does not support cluster setup with more than one managed instance group: %s", framework.TestContext.CloudConfig.NodeInstanceGroup)
	}
	instanceGroup = framework.TestContext.CloudConfig.NodeInstanceGroup

	// Recreate the nodes.
	for zone, nodeNames := range nodeNamesByZone {
		args := []string{
			"compute",
			fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
			"instance-groups",
			"managed",
			"recreate-instances",
			instanceGroup,
		}

		args = append(args, fmt.Sprintf("--instances=%s", strings.Join(nodeNames, ",")))
		args = append(args, fmt.Sprintf("--zone=%s", zone))
		e2elog.Logf("Recreating instance group %s.", instanceGroup)
		stdout, stderr, err := framework.RunCmd("gcloud", args...)
		if err != nil {
			return fmt.Errorf("error recreating nodes: %s\nstdout: %s\nstderr: %s", err, stdout, stderr)
		}
	}
	return nil
}

// WaitForNodeBootIdsToChange waits for the boot ids of the given nodes to change in order to verify the node has been recreated.
func WaitForNodeBootIdsToChange(c clientset.Interface, nodes []v1.Node, timeout time.Duration) error {
	errMsg := []string{}
	for i := range nodes {
		node := &nodes[i]
		if err := wait.Poll(30*time.Second, timeout, func() (bool, error) {
			newNode, err := c.CoreV1().Nodes().Get(node.Name, metav1.GetOptions{})
			if err != nil {
				e2elog.Logf("Could not get node info: %s. Retrying in %v.", err, 30*time.Second)
				return false, nil
			}
			return node.Status.NodeInfo.BootID != newNode.Status.NodeInfo.BootID, nil
		}); err != nil {
			errMsg = append(errMsg, "Error waiting for node %s boot ID to change: %s", node.Name, err.Error())
		}
	}
	if len(errMsg) > 0 {
		return fmt.Errorf(strings.Join(errMsg, ","))
	}
	return nil
}
