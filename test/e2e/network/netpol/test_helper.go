/*
Copyright 2020 The Kubernetes Authors.

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

package netpol

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	"sigs.k8s.io/yaml"
)

const (
	waitInterval = 1 * time.Second
	waitTimeout  = 30 * time.Second
)

// prettyPrint a networkPolicy
func prettyPrint(policy *networkingv1.NetworkPolicy) string {
	raw, err := yaml.Marshal(policy)
	framework.ExpectNoError(err, "marshal network policy to yaml")
	return string(raw)
}

// CreatePolicy creates a policy in the given namespace
func CreatePolicy(k8s *kubeManager, policy *networkingv1.NetworkPolicy, namespace string) {
	if isVerbose {
		framework.Logf("****************************************************************")
		framework.Logf("Network Policy creating %s/%s \n%s", namespace, policy.Name, prettyPrint(policy))
		framework.Logf("****************************************************************")
	}

	_, err := k8s.createNetworkPolicy(namespace, policy)
	framework.ExpectNoError(err, "Unable to create netpol %s/%s", namespace, policy.Name)
}

// UpdatePolicy updates a networkpolicy
func UpdatePolicy(k8s *kubeManager, policy *networkingv1.NetworkPolicy, namespace string) {
	if isVerbose {
		framework.Logf("****************************************************************")
		framework.Logf("Network Policy updating %s/%s \n%s", namespace, policy.Name, prettyPrint(policy))
		framework.Logf("****************************************************************")
	}

	_, err := k8s.updateNetworkPolicy(namespace, policy)
	framework.ExpectNoError(err, "Unable to update netpol %s/%s", namespace, policy.Name)
}

// waitForHTTPServers waits for all webservers to be up, on all protocols sent in the input,  and then validates them using the same probe logic as the rest of the suite.
func waitForHTTPServers(k *kubeManager, model *Model) error {
	const maxTries = 10
	framework.Logf("waiting for HTTP servers (ports 80 and/or 81) to become ready")

	testCases := map[string]*TestCase{}
	for _, port := range model.Ports {
		// Protocols is provided as input so that we can skip udp polling for windows
		for _, protocol := range model.Protocols {
			fromPort := 81
			desc := fmt.Sprintf("%d->%d,%s", fromPort, port, protocol)
			testCases[desc] = &TestCase{ToPort: int(port), Protocol: protocol}
		}
	}
	notReady := map[string]bool{}
	for caseName := range testCases {
		notReady[caseName] = true
	}

	for i := 0; i < maxTries; i++ {
		for caseName, testCase := range testCases {
			if notReady[caseName] {
				reachability := NewReachability(model.AllPods(), true)
				testCase.Reachability = reachability
				ProbePodToPodConnectivity(k, model, testCase)
				_, wrong, _, _ := reachability.Summary(ignoreLoopback)
				if wrong == 0 {
					framework.Logf("server %s is ready", caseName)
					delete(notReady, caseName)
				} else {
					framework.Logf("server %s is not ready", caseName)
				}
			}
		}
		if len(notReady) == 0 {
			return nil
		}
		time.Sleep(waitInterval)
	}
	return fmt.Errorf("after %d tries, %d HTTP servers are not ready", maxTries, len(notReady))
}

// ValidateOrFail validates connectivity
func ValidateOrFail(k8s *kubeManager, model *Model, testCase *TestCase) {
	ginkgo.By("Validating reachability matrix...")

	// 1st try
	ginkgo.By("Validating reachability matrix... (FIRST TRY)")
	ProbePodToPodConnectivity(k8s, model, testCase)
	// 2nd try, in case first one failed
	if _, wrong, _, _ := testCase.Reachability.Summary(ignoreLoopback); wrong != 0 {
		framework.Logf("failed first probe %d wrong results ... retrying (SECOND TRY)", wrong)
		ProbePodToPodConnectivity(k8s, model, testCase)
	}

	// at this point we know if we passed or failed, print final matrix and pass/fail the test.
	if _, wrong, _, _ := testCase.Reachability.Summary(ignoreLoopback); wrong != 0 {
		testCase.Reachability.PrintSummary(true, true, true)
		framework.Failf("Had %d wrong results in reachability matrix", wrong)
	}
	if isVerbose {
		testCase.Reachability.PrintSummary(true, true, true)
	}
	framework.Logf("VALIDATION SUCCESSFUL")
}

// UpdateNamespaceLabels sets the labels for a namespace
func UpdateNamespaceLabels(k8s *kubeManager, ns string, newNsLabel map[string]string) {
	err := k8s.setNamespaceLabels(ns, newNsLabel)
	framework.ExpectNoError(err, "Update namespace %s labels", ns)
	err = wait.PollImmediate(waitInterval, waitTimeout, func() (done bool, err error) {
		namespace, err := k8s.getNamespace(ns)
		if err != nil {
			return false, err
		}
		for key, expected := range newNsLabel {
			if actual, ok := namespace.Labels[key]; !ok || (expected != actual) {
				return false, nil
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Unable to wait for ns %s to update labels", ns)
}

// AddPodLabels adds new labels to a deployment's template
func AddPodLabels(k8s *kubeManager, pod *Pod, newPodLabels map[string]string) {
	kubePod, err := k8s.clientSet.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Unable to get pod %s/%s", pod.Namespace, pod.Name)
	if kubePod.Labels == nil {
		kubePod.Labels = map[string]string{}
	}
	for key, val := range newPodLabels {
		kubePod.Labels[key] = val
	}
	_, err = k8s.clientSet.CoreV1().Pods(pod.Namespace).Update(context.TODO(), kubePod, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "Unable to add pod %s/%s labels", pod.Namespace, pod.Name)

	err = wait.PollImmediate(waitInterval, waitTimeout, func() (done bool, err error) {
		waitForPod, err := k8s.getPod(pod.Namespace, pod.Name)
		if err != nil {
			return false, err
		}
		for key, expected := range newPodLabels {
			if actual, ok := waitForPod.Labels[key]; !ok || (expected != actual) {
				return false, nil
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Unable to wait for pod %s/%s to update labels", pod.Namespace, pod.Name)
}

// ResetNamespaceLabels resets the labels for a namespace
func ResetNamespaceLabels(k8s *kubeManager, ns string) {
	UpdateNamespaceLabels(k8s, ns, (&Namespace{Name: ns}).LabelSelector())
}

// ResetPodLabels resets the labels for a deployment's template
func ResetPodLabels(k8s *kubeManager, pod *Pod) {
	kubePod, err := k8s.clientSet.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Unable to get pod %s/%s", pod.Namespace, pod.Name)
	kubePod.Labels = pod.LabelSelector()
	_, err = k8s.clientSet.CoreV1().Pods(pod.Namespace).Update(context.TODO(), kubePod, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "Unable to add pod %s/%s labels", pod.Namespace, pod.Name)

	err = wait.PollImmediate(waitInterval, waitTimeout, func() (done bool, err error) {
		waitForPod, err := k8s.getPod(pod.Namespace, pod.Name)
		if err != nil {
			return false, nil
		}
		for key, expected := range pod.LabelSelector() {
			if actual, ok := waitForPod.Labels[key]; !ok || (expected != actual) {
				return false, nil
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Unable to wait for pod %s/%s to update labels", pod.Namespace, pod.Name)
}
