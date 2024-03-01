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
	"context"
	"time"

	"golang.org/x/sys/unix"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
)

const contentionLockFile = "/var/run/kubelet.lock"

// Kubelet Lock contention tests the lock contention feature.
// Disruptive because the kubelet is restarted in the test.
// NodeSpecialFeature:LockContention because we don't want the test to be picked up by any other
// test suite, hence the unique name "LockContention".
var _ = SIGDescribe("Lock contention", framework.WithSlow(), framework.WithDisruptive(), "[NodeSpecialFeature:LockContention]", func() {

	// Requires `--lock-file` & `--exit-on-lock-contention` flags to be set on the Kubelet.
	ginkgo.It("Kubelet should stop when the test acquires the lock on lock file and restart once the lock is released", func(ctx context.Context) {

		ginkgo.By("perform kubelet health check to check if kubelet is healthy and running.")
		// Precautionary check that kubelet is healthy before running the test.
		gomega.Expect(kubeletHealthCheck(kubeletHealthCheckURL)).To(gomega.BeTrueBecause("expected kubelet to be in healthy state"))

		ginkgo.By("acquiring the lock on lock file i.e /var/run/kubelet.lock")
		// Open the file with the intention to acquire the lock, this would imitate the behaviour
		// of the another kubelet(self-hosted) trying to start. When this lock contention happens
		// it is expected that the running kubelet must terminate and wait until the lock on the
		// lock file is released.
		// Kubelet uses the same approach to acquire the lock on lock file as shown here:
		// https://github.com/kubernetes/kubernetes/blob/9d2b361ebc7ef28f7cb75596ef40b7c239732d37/cmd/kubelet/app/server.go#L512-#L523
		// and the function definition of Acquire is here:
		// https://github.com/kubernetes/kubernetes/blob/9d2b361ebc7ef28f7cb75596ef40b7c239732d37/pkg/util/flock/flock_unix.go#L26
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
		// It should not be as the lock contention forces the kubelet to stop.
		gomega.Eventually(ctx, func() bool {
			return kubeletHealthCheck(kubeletHealthCheckURL)
		}, 10*time.Second, time.Second).Should(gomega.BeFalseBecause("expected kubelet to not be in healthy state"))
	})
})
