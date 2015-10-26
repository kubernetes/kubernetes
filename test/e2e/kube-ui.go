/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"net/http"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("kube-ui", func() {
	const (
		uiServiceName = "kube-ui"
		uiAppName     = uiServiceName
		uiNamespace   = api.NamespaceSystem

		serverStartTimeout = 1 * time.Minute
	)

	f := NewFramework("kube-ui")

	It("should check that the kube-ui instance is alive [Conformance]", func() {
		By("Checking the kube-ui service exists.")
		err := waitForService(f.Client, uiNamespace, uiServiceName, true, poll, serviceStartTimeout)
		Expect(err).NotTo(HaveOccurred())

		By("Checking to make sure the kube-ui pods are running")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"k8s-app": uiAppName}))
		err = waitForPodsWithLabelRunning(f.Client, uiNamespace, selector)
		Expect(err).NotTo(HaveOccurred())

		By("Checking to make sure we get a response from the kube-ui.")
		err = wait.Poll(poll, serverStartTimeout, func() (bool, error) {
			var status int
			// Query against the proxy URL for the kube-ui service.
			err := f.Client.Get().
				Namespace(uiNamespace).
				Prefix("proxy").
				Resource("services").
				Name(uiServiceName).
				Timeout(singleCallTimeout).
				Do().
				StatusCode(&status).
				Error()
			if status != http.StatusOK {
				Logf("Unexpected status from kube-ui: %v", status)
			} else if err != nil {
				Logf("Request to kube-ui failed: %v", err)
			}
			// Don't return err here as it aborts polling.
			return status == http.StatusOK, nil
		})
		Expect(err).NotTo(HaveOccurred())

		By("Checking that the ApiServer /ui endpoint redirects to a valid server.")
		var status int
		err = f.Client.Get().
			AbsPath("/ui").
			Timeout(singleCallTimeout).
			Do().
			StatusCode(&status).
			Error()
		Expect(err).NotTo(HaveOccurred())
		Expect(status).To(Equal(http.StatusOK), "Unexpected status from /ui")
	})
})
