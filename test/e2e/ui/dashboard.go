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

package ui

import (
	"context"
	"net/http"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = SIGDescribe("Kubernetes Dashboard", func() {
	const (
		uiServiceName = "kubernetes-dashboard"
		uiAppName     = uiServiceName
		uiNamespace   = metav1.NamespaceSystem

		serverStartTimeout = 1 * time.Minute
	)

	f := framework.NewDefaultFramework(uiServiceName)

	It("should check that the kubernetes-dashboard instance is alive", func() {
		By("Checking whether the kubernetes-dashboard service exists.")
		err := framework.WaitForService(f.ClientSet, uiNamespace, uiServiceName, true, framework.Poll, framework.ServiceStartTimeout)
		Expect(err).NotTo(HaveOccurred())

		By("Checking to make sure the kubernetes-dashboard pods are running")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": uiAppName}))
		err = testutils.WaitForPodsWithLabelRunning(f.ClientSet, uiNamespace, selector)
		Expect(err).NotTo(HaveOccurred())

		By("Checking to make sure we get a response from the kubernetes-dashboard.")
		err = wait.Poll(framework.Poll, serverStartTimeout, func() (bool, error) {
			var status int
			proxyRequest, errProxy := framework.GetServicesProxyRequest(f.ClientSet, f.ClientSet.CoreV1().RESTClient().Get())
			if errProxy != nil {
				framework.Logf("Get services proxy request failed: %v", errProxy)
			}

			ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
			defer cancel()

			// Query against the proxy URL for the kube-ui service.
			err := proxyRequest.Namespace(uiNamespace).
				Context(ctx).
				Name(uiServiceName).
				Timeout(framework.SingleCallTimeout).
				Do().
				StatusCode(&status).
				Error()
			if err != nil {
				if ctx.Err() != nil {
					framework.Failf("Request to kube-ui failed: %v", err)
					return true, err
				}
				framework.Logf("Request to kube-ui failed: %v", err)
			} else if status != http.StatusOK {
				framework.Logf("Unexpected status from kubernetes-dashboard: %v", status)
			}
			// Don't return err here as it aborts polling.
			return status == http.StatusOK, nil
		})
		Expect(err).NotTo(HaveOccurred())

		By("Checking that the ApiServer /ui endpoint redirects to a valid server.")
		var status int
		err = f.ClientSet.CoreV1().RESTClient().Get().
			AbsPath("/ui").
			Timeout(framework.SingleCallTimeout).
			Do().
			StatusCode(&status).
			Error()
		Expect(err).NotTo(HaveOccurred())
		Expect(status).To(Equal(http.StatusOK), "Unexpected status from /ui")
	})
})
