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
func CreatePolicy(ctx context.Context, k8s *kubeManager, policy *networkingv1.NetworkPolicy, namespace string) {
	if isVerbose {
		framework.Logf("****************************************************************")
		framework.Logf("Network Policy creating %s/%s \n%s", namespace, policy.Name, prettyPrint(policy))
		framework.Logf("****************************************************************")
	}

	_, err := k8s.createNetworkPolicy(ctx, namespace, policy)
	framework.ExpectNoError(err, "Unable to create netpol %s/%s", namespace, policy.Name)
}

// UpdatePolicy updates a networkpolicy
func UpdatePolicy(ctx context.Context, k8s *kubeManager, policy *networkingv1.NetworkPolicy, namespace string) {
	if isVerbose {
		framework.Logf("****************************************************************")
		framework.Logf("Network Policy updating %s/%s \n%s", namespace, policy.Name, prettyPrint(policy))
		framework.Logf("****************************************************************")
	}

	_, err := k8s.updateNetworkPolicy(ctx, namespace, policy)
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
				reachability := NewReachability(k.AllPodStrings(), true)
				testCase.Reachability = reachability
				ProbePodToPodConnectivity(k, k.AllPods(), k.DNSDomain(), testCase)
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
func ValidateOrFail(k8s *kubeManager, testCase *TestCase) {
	ginkgo.By("Validating reachability matrix...")

	// 1st try, exponential backoff (starting at 1s) will happen for every probe to accommodate infra that might be
	// network-congested, as is common in some GH actions or other heavily oversubscribed CI systems.
	ginkgo.By("Validating reachability matrix... (FIRST TRY)")
	ProbePodToPodConnectivity(k8s, k8s.AllPods(), k8s.DNSDomain(), testCase)

	// the aforementioned individual probe's exponential retries (introduced in january 2023) might be able to obviate
	//  this step, let's investigate removing this massive secondary polling of the matrix some day.
	if _, wrong, _, _ := testCase.Reachability.Summary(ignoreLoopback); wrong != 0 {
		framework.Logf("failed first probe %d wrong results ... retrying (SECOND TRY)", wrong)
		ProbePodToPodConnectivity(k8s, k8s.AllPods(), k8s.DNSDomain(), testCase)
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

// AddNamespaceLabels adds a new label to a namespace
func AddNamespaceLabel(ctx context.Context, k8s *kubeManager, name string, key string, val string) {
	ns, err := k8s.getNamespace(ctx, name)
	framework.ExpectNoError(err, "Unable to get namespace %s", name)
	ns.Labels[key] = val
	_, err = k8s.clientSet.CoreV1().Namespaces().Update(ctx, ns, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "Unable to update namespace %s", name)
}

// DeleteNamespaceLabel deletes a label from a namespace (if present)
func DeleteNamespaceLabel(ctx context.Context, k8s *kubeManager, name string, key string) {
	ns, err := k8s.getNamespace(ctx, name)
	framework.ExpectNoError(err, "Unable to get namespace %s", name)
	if _, ok := ns.Labels[key]; !ok {
		// nothing to do if the label is not present
		return
	}
	delete(ns.Labels, key)
	_, err = k8s.clientSet.CoreV1().Namespaces().Update(ctx, ns, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "Unable to update namespace %s", name)
}

// AddPodLabels adds new labels to a running pod
func AddPodLabels(ctx context.Context, k8s *kubeManager, namespace string, name string, newPodLabels map[string]string) {
	kubePod, err := k8s.clientSet.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Unable to get pod %s/%s", namespace, name)
	if kubePod.Labels == nil {
		kubePod.Labels = map[string]string{}
	}
	for key, val := range newPodLabels {
		kubePod.Labels[key] = val
	}
	_, err = k8s.clientSet.CoreV1().Pods(namespace).Update(ctx, kubePod, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "Unable to add pod %s/%s labels", namespace, name)

	err = wait.PollUntilContextTimeout(ctx, waitInterval, waitTimeout, true, func(ctx context.Context) (done bool, err error) {
		waitForPod, err := k8s.getPod(ctx, namespace, name)
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
	framework.ExpectNoError(err, "Unable to wait for pod %s/%s to update labels", namespace, name)
}

// ResetPodLabels resets the labels for a deployment's template
func ResetPodLabels(ctx context.Context, k8s *kubeManager, namespace string, name string) {
	kubePod, err := k8s.clientSet.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Unable to get pod %s/%s", namespace, name)
	labels := map[string]string{
		podNameLabelKey(): name,
	}
	kubePod.Labels = labels
	_, err = k8s.clientSet.CoreV1().Pods(namespace).Update(ctx, kubePod, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "Unable to add pod %s/%s labels", namespace, name)

	err = wait.PollUntilContextTimeout(ctx, waitInterval, waitTimeout, true, func(ctx context.Context) (done bool, err error) {
		waitForPod, err := k8s.getPod(ctx, namespace, name)
		if err != nil {
			return false, nil
		}
		for key, expected := range labels {
			if actual, ok := waitForPod.Labels[key]; !ok || (expected != actual) {
				return false, nil
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Unable to wait for pod %s/%s to update labels", namespace, name)
}
