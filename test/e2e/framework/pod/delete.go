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

package pod

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// PodDeleteTimeout is how long to wait for a pod to be deleted.
	PodDeleteTimeout = 5 * time.Minute
)

// DeletePodOrFail deletes the pod of the specified namespace and name. Resilient to the pod
// not existing.
func DeletePodOrFail(ctx context.Context, c clientset.Interface, ns, name string) {
	ginkgo.By(fmt.Sprintf("Deleting pod %s in namespace %s", name, ns))
	err := c.CoreV1().Pods(ns).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && apierrors.IsNotFound(err) {
		return
	}

	framework.ExpectNoErrorWithOffset(1, err, "failed to delete pod %s in namespace %s", name, ns)
}

// DeletePodWithWait deletes the passed-in pod and waits for the pod to be terminated. Resilient to the pod
// not existing.
func DeletePodWithWait(ctx context.Context, c clientset.Interface, pod *v1.Pod) error {
	if pod == nil {
		return nil
	}
	return DeletePodWithWaitByName(ctx, c, pod.GetName(), pod.GetNamespace())
}

// DeletePodWithWaitByName deletes the named and namespaced pod and waits for the pod to be terminated. Resilient to the pod
// not existing.
func DeletePodWithWaitByName(ctx context.Context, c clientset.Interface, podName, podNamespace string) error {
	framework.Logf("Deleting pod %q in namespace %q", podName, podNamespace)
	err := c.CoreV1().Pods(podNamespace).Delete(ctx, podName, metav1.DeleteOptions{})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil // assume pod was already deleted
		}
		return fmt.Errorf("pod Delete API error: %w", err)
	}
	framework.Logf("Wait up to %v for pod %q to be fully deleted", PodDeleteTimeout, podName)
	err = WaitForPodNotFoundInNamespace(ctx, c, podName, podNamespace, PodDeleteTimeout)
	if err != nil {
		return fmt.Errorf("pod %q was not deleted: %w", podName, err)
	}
	return nil
}

// DeletePodWithGracePeriod deletes the passed-in pod. Resilient to the pod not existing.
func DeletePodWithGracePeriod(ctx context.Context, c clientset.Interface, pod *v1.Pod, grace int64) error {
	return DeletePodWithGracePeriodByName(ctx, c, pod.GetName(), pod.GetNamespace(), grace)
}

// DeletePodsWithGracePeriod deletes the passed-in pods. Resilient to the pods not existing.
func DeletePodsWithGracePeriod(ctx context.Context, c clientset.Interface, pods []v1.Pod, grace int64) error {
	for _, pod := range pods {
		if err := DeletePodWithGracePeriod(ctx, c, &pod, grace); err != nil {
			return err
		}
	}
	return nil
}

// DeletePodWithGracePeriodByName deletes a pod by name and namespace. Resilient to the pod not existing.
func DeletePodWithGracePeriodByName(ctx context.Context, c clientset.Interface, podName, podNamespace string, grace int64) error {
	framework.Logf("Deleting pod %q in namespace %q", podName, podNamespace)
	err := c.CoreV1().Pods(podNamespace).Delete(ctx, podName, *metav1.NewDeleteOptions(grace))
	if err != nil {
		if apierrors.IsNotFound(err) {
			return nil // assume pod was already deleted
		}
		return fmt.Errorf("pod Delete API error: %w", err)
	}
	return nil
}
