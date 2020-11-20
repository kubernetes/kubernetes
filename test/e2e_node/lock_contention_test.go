// +build linux

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

package e2enode

import (
	"os"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
)

var contentionFile = "/var/run/kubelet.lock"

var _ = SIGDescribe("Lock contention [Slow] [Disruptive] [NodeFeature:LockContention]", func() {
	ginkgo.It("should stop kubelet when the contention file is created.", func() {
		ginkgo.By("create the lock contention file")

		_, err := os.Create(contentionFile)
		framework.ExpectNoError(err)

		ginkgo.By("verifying the kubelet is not running anymore.")
		gomega.Eventually(func() string {
			_, kubeletService := kubeletRunningStatus()
			return kubeletService
		}, 10*time.Second, time.Second).Should(gomega.Not(gomega.HavePrefix("kubelet-")))
	})
})
