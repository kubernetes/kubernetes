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

package e2e

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	k8sAppKey    = "k8s-app"
	esValue      = "elasticsearch-logging"
	fluentdValue = "fluentd-logging"

	synthLoggerContainerName = "synth-logger"

	// ingestionTimeout is how long to keep retrying to wait for all the
	// logs to be ingested.
	ingestionTimeout = 10 * time.Minute

	// countTo is the number of log lines emitted (and checked) for each synthetic logging pod.
	countTo = 100
)

func bodyToJsonObject(body []byte) (map[string]interface{}, error) {
	var r map[string]interface{}
	if err := json.Unmarshal(body, &r); err != nil {
		framework.Logf("Bad JSON: %s", string(body))
		return nil, fmt.Errorf("failed to unmarshal json: %v", err)
	}
	return r, nil
}

func nodeInNodeList(nodeName string, nodeList *api.NodeList) bool {
	for _, node := range nodeList.Items {
		if nodeName == node.Name {
			return true
		}
	}
	return false
}

func cleanupLoggingPods(f *framework.Framework, podNames []string) {
	for _, pod := range podNames {
		if err := f.Client.Pods(f.Namespace.Name).Delete(pod, nil); err != nil {
			framework.Logf("Failed to delete pod %s: %v", pod, err)
		}
	}
}

func waitForPodsToSucceed(f *framework.Framework, podNames []string) error {
	for _, pod := range podNames {
		err := framework.WaitForPodSuccessInNamespace(f.Client, pod, synthLoggerContainerName, f.Namespace.Name)
		if err != nil {
			return err
		}
	}

	return nil
}

func createSynthLoggers(f *framework.Framework, nodes *api.NodeList) (taintName string, podNames []string, _ error) {
	// Create a unique root name for the resources in this test to permit
	// parallel executions of this test.
	// Use a unique namespace for the resources created in this test.
	ns := f.Namespace.Name
	name := "synthlogger"

	// Form a unique name to taint log lines to be collected.
	// Replace '-' characters with '_' to prevent the analyzer from breaking apart names.
	taintName = strings.Replace(ns+name, "-", "_", -1)
	framework.Logf("Tainting log lines with %v", taintName)

	// podNames records the names of the synthetic logging pods that are created in the loop below.
	// Instantiate a synthetic logger pod on each node.
	for i, node := range nodes.Items {
		podName := fmt.Sprintf("%s-%d", name, i)
		_, err := f.Client.Pods(ns).Create(&api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": name},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  synthLoggerContainerName,
						Image: "gcr.io/google_containers/ubuntu:14.04",
						// notice: the subshell syntax is escaped with `$$`
						Command: []string{"bash", "-c", fmt.Sprintf("i=0; while ((i < %d)); do echo \"%d %s $i %s\"; i=$$(($i+1)); done", countTo, i, taintName, podName)},
					},
				},
				NodeName:      node.Name,
				RestartPolicy: api.RestartPolicyNever,
			},
		})
		if err != nil {
			return "", nil, err
		}
		podNames = append(podNames, podName)
	}

	return
}

func getFluentdPods(f *framework.Framework) (*api.PodList, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{k8sAppKey: fluentdValue}))
	options := api.ListOptions{LabelSelector: label}
	return f.Client.Pods(api.NamespaceSystem).List(options)
}

func waitForFluentdPods(f *framework.Framework, nodes *api.NodeList, fluentdPods *api.PodList) error {
	// Wait for fluentd pods to become running
	for _, pod := range fluentdPods.Items {
		if nodeInNodeList(pod.Spec.NodeName, nodes) {
			if err := framework.WaitForPodRunningInNamespace(f.Client, &pod); err != nil {
				return err
			}
		}
	}

	// Check if each healthy node has fluentd running on it
	for _, node := range nodes.Items {
		exists := false
		for _, pod := range fluentdPods.Items {
			if pod.Spec.NodeName == node.Name {
				exists = true
				break
			}
		}
		if !exists {
			return fmt.Errorf("Node %v does not have fluentd pod running on it.", node.Name)
		}
	}

	return nil
}

func getHealthyNodes(f *framework.Framework) (nodes *api.NodeList) {
	nodes = framework.GetReadySchedulableNodesOrDie(f.Client)
	if len(nodes.Items) == 0 {
		framework.Failf("Failed to find any nodes")
	}
	framework.Logf("Found %d nodes.", len(nodes.Items))

	// Filter out unhealthy nodes.
	// Previous tests may have cause failures of some nodes. Let's skip
	// 'Not Ready' nodes, just in case (there is no need to fail the test).
	framework.FilterNodes(nodes, func(node api.Node) bool {
		return framework.IsNodeConditionSetAsExpected(&node, api.NodeReady, true)
	})
	if len(nodes.Items) < 2 {
		framework.Failf("Less than two nodes were found Ready: %d", len(nodes.Items))
	}
	framework.Logf("Found %d healthy nodes.", len(nodes.Items))

	return
}
