//go:build linux
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
	"time"

	"golang.org/x/sys/unix"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
)

const contentionLockFile = "/var/run/kubelet.lock"

var _ = SIGDescribe("Lock contention [Slow] [Disruptive] [Serial] [NodeFeature:LockContention]", func() {

	ginkgo.It("Kubelet should stop when the test acquires the lock on lock file and restart once the lock is released", func() {

		ginkgo.By("perform kubelet health check to check if kubelet is healthy and running.")
		// Precautionary check that kubelet is healthy before running the test.
		gomega.Expect(kubeletHealthCheck(kubeletHealthCheckURL)).To(gomega.BeTrue())

		ginkgo.By("acquiring the lock on lock file i.e /var/run/kubelet.lock")
		// Open the file with the intention to acquire the lock, this would imitate the behaviour
		// of the another kubelet(self-hosted) trying to start. When this lock contention happens
		// it is expected that the running kubelet must terminate and wait until the lock on the
		// lock file is released.
		// Kubelet uses the same approach to acquire the lock on lock file as shown here:
		// https://github.com/kubernetes/kubernetes/blob/master/cmd/kubelet/app/server.go#L530-#L546
		// and the function definition of Acquire is here:
		// https://github.com/kubernetes/kubernetes/blob/master/pkg/util/flock/flock_unix.go#L25
		fd, err := unix.Open(contentionLockFile, unix.O_CREAT|unix.O_RDWR|unix.O_CLOEXEC, 0600)
		framework.ExpectNoError(err)
		// Defer the lock release in case test fails and we don't reach the step of the release
		// lock. This ensures that we release the lock for sure.
		defer func() {
			err = unix.Flock(fd, unix.LOCK_UN)
			framework.ExpectNoError(err)
		}()
		// Acquire lock.
		err = unix.Flock(fd, unix.LOCK_EX)
		framework.ExpectNoError(err)

		ginkgo.By("verifying the kubelet is not healthy as there was a lock contention.")
		// Once the lock is acquired, check if the kubelet is in healthy state or not.
		// It should not be.
		gomega.Eventually(func() bool {
			return kubeletHealthCheck(kubeletHealthCheckURL)
		}, 10*time.Second, time.Second).Should(gomega.BeFalse())

		ginkgo.By("releasing the lock on lock file i.e /var/run/kubelet.lock")
		// Release the lock.
		err = unix.Flock(fd, unix.LOCK_UN)
		framework.ExpectNoError(err)

		ginkgo.By("verifying the kubelet is healthy again after the lock was released.")
		// Releasing the lock triggers kubelet to re-acquire the lock and restart.
		// Hence the kubelet should report healthy state.
		gomega.Eventually(func() bool {
			return kubeletHealthCheck(kubeletHealthCheckURL)
		}, 10*time.Second, time.Second).Should(gomega.BeTrue())
	})
})
