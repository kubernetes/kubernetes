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

package service

import (
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	servicehelper "k8s.io/cloud-provider/service/helpers"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo/v2"
)

// WaitForServiceDeletedWithFinalizer waits for the service with finalizer to be deleted.
func WaitForServiceDeletedWithFinalizer(ctx context.Context, cs clientset.Interface, namespace, name string) {
	ginkgo.By("Delete service with finalizer")
	if err := cs.CoreV1().Services(namespace).Delete(ctx, name, metav1.DeleteOptions{}); err != nil {
		framework.Failf("Failed to delete service %s/%s", namespace, name)
	}

	ginkgo.By("Wait for service to disappear")
	if pollErr := wait.PollUntilContextTimeout(ctx, LoadBalancerPollInterval, GetServiceLoadBalancerCreationTimeout(ctx, cs), true, func(ctx context.Context) (bool, error) {
		svc, err := cs.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				framework.Logf("Service %s/%s is gone.", namespace, name)
				return true, nil
			}
			return false, err
		}
		framework.Logf("Service %s/%s still exists with finalizers: %v", namespace, name, svc.Finalizers)
		return false, nil
	}); pollErr != nil {
		framework.Failf("Failed to wait for service to disappear: %v", pollErr)
	}
}

// WaitForServiceUpdatedWithFinalizer waits for the service to be updated to have or
// don't have a finalizer.
func WaitForServiceUpdatedWithFinalizer(ctx context.Context, cs clientset.Interface, namespace, name string, hasFinalizer bool) {
	ginkgo.By(fmt.Sprintf("Wait for service to hasFinalizer=%t", hasFinalizer))
	if pollErr := wait.PollUntilContextTimeout(ctx, LoadBalancerPollInterval, GetServiceLoadBalancerCreationTimeout(ctx, cs), true, func(ctx context.Context) (bool, error) {
		svc, err := cs.CoreV1().Services(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		foundFinalizer := false
		for _, finalizer := range svc.Finalizers {
			if finalizer == servicehelper.LoadBalancerCleanupFinalizer {
				foundFinalizer = true
				break
			}
		}
		if foundFinalizer != hasFinalizer {
			framework.Logf("Service %s/%s hasFinalizer=%t, want %t", namespace, name, foundFinalizer, hasFinalizer)
			return false, nil
		}
		return true, nil
	}); pollErr != nil {
		framework.Failf("Failed to wait for service to hasFinalizer=%t: %v", hasFinalizer, pollErr)
	}
}
