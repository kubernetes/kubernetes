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

package utils

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/utils/integer"
)

// EnsureLoggingAgentDeployment checks that logging agent is present on each
// node and returns an error if that's not true.
func EnsureLoggingAgentDeployment(f *framework.Framework, appName string) error {
	agentPods, err := getLoggingAgentPods(f, appName)
	if err != nil {
		return fmt.Errorf("failed to get logging agent pods: %v", err)
	}

	agentPerNode := make(map[string]int)
	for _, pod := range agentPods.Items {
		agentPerNode[pod.Spec.NodeName]++
	}

	nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
	for _, node := range nodeList.Items {
		agentPodsCount, ok := agentPerNode[node.Name]

		if !ok {
			return fmt.Errorf("node %s doesn't have logging agents, want 1", node.Name)
		} else if agentPodsCount != 1 {
			return fmt.Errorf("node %s has %d logging agents, want 1", node.Name, agentPodsCount)
		}
	}

	return nil
}

// EnsureLoggingAgentRestartsCount checks that each logging agent was restarted
// no more than maxRestarts times and returns an error if there's a pod which
// exceeds this number of restarts.
func EnsureLoggingAgentRestartsCount(f *framework.Framework, appName string, maxRestarts int) error {
	agentPods, err := getLoggingAgentPods(f, appName)
	if err != nil {
		return fmt.Errorf("failed to get logging agent pods: %v", err)
	}

	maxRestartCount := 0
	for _, pod := range agentPods.Items {
		contStatuses := pod.Status.ContainerStatuses
		if len(contStatuses) == 0 {
			e2elog.Logf("There are no container statuses for pod %s", pod.Name)
			continue
		}
		restartCount := int(contStatuses[0].RestartCount)
		maxRestartCount = integer.IntMax(maxRestartCount, restartCount)

		e2elog.Logf("Logging agent %s on node %s was restarted %d times",
			pod.Name, pod.Spec.NodeName, restartCount)
	}

	if maxRestartCount > maxRestarts {
		return fmt.Errorf("max logging agent restarts was %d, which is more than allowed %d",
			maxRestartCount, maxRestarts)
	}
	return nil
}

func getLoggingAgentPods(f *framework.Framework, appName string) (*v1.PodList, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": appName}))
	options := meta_v1.ListOptions{LabelSelector: label.String()}
	return f.ClientSet.CoreV1().Pods(api.NamespaceSystem).List(options)
}
