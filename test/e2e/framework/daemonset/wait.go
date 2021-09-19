/*
Copyright 2021 The Kubernetes Authors.

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

package daemonset

import (
	"context"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
)

// WaitForDaemonSetAvailable waits up to timeout for all pods of the specified
// daemonset to become available.
func WaitForDaemonSetAvailable(c clientset.Interface, ns string, name string, timeout time.Duration) error {
	start := time.Now()

	return wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		daemonset, err := c.AppsV1().DaemonSets(ns).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Error getting daemonset %s in namespace %s: %v", name, ns, err)
			return false, err
		}

		framework.Logf("%d / %d pods available and %d / %d pods up-to-date for daemonset %s in namespace %s (%d seconds elapsed)",
			daemonset.Status.NumberAvailable, daemonset.Status.DesiredNumberScheduled,
			daemonset.Status.UpdatedNumberScheduled, daemonset.Status.DesiredNumberScheduled,
			name, ns, int(time.Since(start).Seconds()))

		allPodsAreAvailable := daemonset.Status.NumberAvailable == daemonset.Status.DesiredNumberScheduled
		allPodsAreUpToDate := daemonset.Status.UpdatedNumberScheduled == daemonset.Status.DesiredNumberScheduled
		return (allPodsAreAvailable && allPodsAreUpToDate), nil
	})
}

// WaitForDaemonSetReady waits up to timeout for all pods of the specified
// daemonset to become ready.
func WaitForDaemonSetReady(c clientset.Interface, ns string, name string, timeout time.Duration) error {
	start := time.Now()

	return wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		daemonset, err := c.AppsV1().DaemonSets(ns).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Error getting daemonset %s in namespace %s: %v", name, ns, err)
			return false, err
		}

		framework.Logf("%d / %d pods ready and %d / %d pods up-to-date for daemonset %s in namespace %s (%d seconds elapsed)",
			daemonset.Status.NumberReady, daemonset.Status.DesiredNumberScheduled,
			daemonset.Status.UpdatedNumberScheduled, daemonset.Status.DesiredNumberScheduled,
			name, ns, int(time.Since(start).Seconds()))

		allPodsAreReady := daemonset.Status.NumberReady == daemonset.Status.DesiredNumberScheduled
		allPodsAreUpToDate := daemonset.Status.UpdatedNumberScheduled == daemonset.Status.DesiredNumberScheduled
		return (allPodsAreReady && allPodsAreUpToDate), nil
	})
}
