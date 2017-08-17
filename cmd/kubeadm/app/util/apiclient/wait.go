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

package apiclient

import (
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// WaitForAPI waits for the API Server's /healthz endpoint to report "ok"
func WaitForAPI(client clientset.Interface, timeout time.Duration) error {
	start := time.Now()
	return wait.PollImmediate(constants.APICallRetryInterval, timeout, func() (bool, error) {
		healthStatus := 0
		client.Discovery().RESTClient().Get().AbsPath("/healthz").Do().StatusCode(&healthStatus)
		if healthStatus != http.StatusOK {
			return false, nil
		}

		fmt.Printf("[apiclient] All control plane components are healthy after %f seconds\n", time.Since(start).Seconds())
		return true, nil
	})
}

// WaitForPodsWithLabel will lookup pods with the given label and wait until they are all
// reporting status as running.
func WaitForPodsWithLabel(client clientset.Interface, timeout time.Duration, out io.Writer, labelKeyValPair string) error {

	lastKnownPodNumber := -1
	return wait.PollImmediate(constants.APICallRetryInterval, timeout, func() (bool, error) {
		listOpts := metav1.ListOptions{LabelSelector: labelKeyValPair}
		pods, err := client.CoreV1().Pods(metav1.NamespaceSystem).List(listOpts)
		if err != nil {
			fmt.Fprintf(out, "[apiclient] Error getting Pods with label selector %q [%v]\n", labelKeyValPair, err)
			return false, nil
		}

		if lastKnownPodNumber != len(pods.Items) {
			fmt.Fprintf(out, "[apiclient] Found %d Pods for label selector %s\n", len(pods.Items), labelKeyValPair)
			lastKnownPodNumber = len(pods.Items)
		}

		if len(pods.Items) == 0 {
			return false, nil
		}

		for _, pod := range pods.Items {
			if pod.Status.Phase != v1.PodRunning {
				return false, nil
			}
		}

		return true, nil
	})
}

// WaitForStaticPodToDisappear blocks until it timeouts or gets a "NotFound" response from the API Server when getting the Static Pod in question
func WaitForStaticPodToDisappear(client clientset.Interface, timeout time.Duration, podName string) error {
	return wait.PollImmediate(constants.APICallRetryInterval, timeout, func() (bool, error) {
		_, err := client.CoreV1().Pods(metav1.NamespaceSystem).Get(podName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			fmt.Printf("[apiclient] The Static Pod %q is now removed\n", podName)
			return true, nil
		}
		return false, nil
	})
}
