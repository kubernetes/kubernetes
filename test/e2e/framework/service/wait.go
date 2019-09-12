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

	"github.com/onsi/ginkgo"
)

// WaitForServiceResponding waits for the service to be responding.
func WaitForServiceResponding(c clientset.Interface, ns, name string) error {
	ginkgo.By(fmt.Sprintf("trying to dial the service %s.%s via the proxy", ns, name))

	return wait.PollImmediate(framework.Poll, RespondingTimeout, func() (done bool, err error) {
		proxyRequest, errProxy := GetServicesProxyRequest(c, c.CoreV1().RESTClient().Get())
		if errProxy != nil {
			framework.Logf("Failed to get services proxy request: %v:", errProxy)
			return false, nil
		}

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		body, err := proxyRequest.Namespace(ns).
			Context(ctx).
			Name(name).
			Do().
			Raw()
		if err != nil {
			if ctx.Err() != nil {
				framework.Failf("Failed to GET from service %s: %v", name, err)
				return true, err
			}
			framework.Logf("Failed to GET from service %s: %v:", name, err)
			return false, nil
		}
		got := string(body)
		if len(got) == 0 {
			framework.Logf("Service %s: expected non-empty response", name)
			return false, err // stop polling
		}
		framework.Logf("Service %s: found nonempty answer: %s", name, got)
		return true, nil
	})
}

// WaitForServiceDeletedWithFinalizer waits for the service with finalizer to be deleted.
func WaitForServiceDeletedWithFinalizer(cs clientset.Interface, namespace, name string) {
	ginkgo.By("Delete service with finalizer")
	if err := cs.CoreV1().Services(namespace).Delete(name, nil); err != nil {
		framework.Failf("Failed to delete service %s/%s", namespace, name)
	}

	ginkgo.By("Wait for service to disappear")
	if pollErr := wait.PollImmediate(LoadBalancerPollInterval, GetServiceLoadBalancerCreationTimeout(cs), func() (bool, error) {
		svc, err := cs.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
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
func WaitForServiceUpdatedWithFinalizer(cs clientset.Interface, namespace, name string, hasFinalizer bool) {
	ginkgo.By(fmt.Sprintf("Wait for service to hasFinalizer=%t", hasFinalizer))
	if pollErr := wait.PollImmediate(LoadBalancerPollInterval, GetServiceLoadBalancerCreationTimeout(cs), func() (bool, error) {
		svc, err := cs.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
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
