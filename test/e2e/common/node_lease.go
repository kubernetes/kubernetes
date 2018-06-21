/*
Copyright 2018 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"time"

	coordv1beta1 "k8s.io/api/coordination/v1beta1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("[Feature:NodeLease][NodeAlphaFeature:NodeLease]", func() {
	f := framework.NewDefaultFramework("node-lease-test")
	Context("when the NodeLease feature is enabled", func() {
		It("the Kubelet should create and update a lease in the kube-node-lease namespace", func() {
			leaseClient := f.ClientSet.CoordinationV1beta1().Leases(corev1.NamespaceNodeLease)
			var (
				err   error
				lease *coordv1beta1.Lease
			)
			// check that lease for this Kubelet exists in the kube-node-lease namespace
			Eventually(func() error {
				lease, err = leaseClient.Get(framework.TestContext.NodeName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				return nil
			}, 5*time.Minute, 5*time.Second).Should(BeNil())
			// check basic expectations for the lease
			Expect(expectLease(lease)).To(BeNil())
			// ensure that at least one lease renewal happens within the
			// lease duration by checking for a change to renew time
			Eventually(func() error {
				newLease, err := leaseClient.Get(framework.TestContext.NodeName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				// check basic expectations for the latest lease
				if err := expectLease(newLease); err != nil {
					return err
				}
				// check that RenewTime has been updated on the latest lease
				newTime := (*newLease.Spec.RenewTime).Time
				oldTime := (*lease.Spec.RenewTime).Time
				if !newTime.After(oldTime) {
					return fmt.Errorf("new lease has time %v, which is not after old lease time %v", newTime, oldTime)
				}
				return nil
			}, time.Duration(*lease.Spec.LeaseDurationSeconds)*time.Second,
				time.Duration(*lease.Spec.LeaseDurationSeconds/3)*time.Second)
		})
	})
})

func expectLease(lease *coordv1beta1.Lease) error {
	// expect values for HolderIdentity, LeaseDurationSeconds, and RenewTime
	if lease.Spec.HolderIdentity == nil {
		return fmt.Errorf("Spec.HolderIdentity should not be nil")
	}
	if lease.Spec.LeaseDurationSeconds == nil {
		return fmt.Errorf("Spec.LeaseDurationSeconds should not be nil")
	}
	if lease.Spec.RenewTime == nil {
		return fmt.Errorf("Spec.RenewTime should not be nil")
	}
	// ensure that the HolderIdentity matches the node name
	if *lease.Spec.HolderIdentity != framework.TestContext.NodeName {
		return fmt.Errorf("Spec.HolderIdentity (%v) should match the node name (%v)", *lease.Spec.HolderIdentity, framework.TestContext.NodeName)
	}
	return nil
}
