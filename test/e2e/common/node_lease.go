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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("NodeLease", func() {
	var nodeName string
	f := framework.NewDefaultFramework("node-lease-test")

	BeforeEach(func() {
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodes.Items)).NotTo(BeZero())
		nodeName = nodes.Items[0].ObjectMeta.Name
	})

	Context("when the NodeLease feature is enabled", func() {
		It("the kubelet should create and update a lease in the kube-node-lease namespace", func() {
			leaseClient := f.ClientSet.CoordinationV1beta1().Leases(corev1.NamespaceNodeLease)
			var (
				err   error
				lease *coordv1beta1.Lease
			)
			By("check that lease for this Kubelet exists in the kube-node-lease namespace")
			Eventually(func() error {
				lease, err = leaseClient.Get(nodeName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				return nil
			}, 5*time.Minute, 5*time.Second).Should(BeNil())
			// check basic expectations for the lease
			Expect(expectLease(lease, nodeName)).To(BeNil())

			By("check that node lease is updated at least once within the lease duration")
			Eventually(func() error {
				newLease, err := leaseClient.Get(nodeName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				// check basic expectations for the latest lease
				if err := expectLease(newLease, nodeName); err != nil {
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
				time.Duration(*lease.Spec.LeaseDurationSeconds/4)*time.Second)
		})

		It("the kubelet should report node status infrequently", func() {
			By("wait until node is ready")
			framework.WaitForNodeToBeReady(f.ClientSet, nodeName, 5*time.Minute)

			By("wait until there is node lease")
			var err error
			var lease *coordv1beta1.Lease
			Eventually(func() error {
				lease, err = f.ClientSet.CoordinationV1beta1().Leases(corev1.NamespaceNodeLease).Get(nodeName, metav1.GetOptions{})
				if err != nil {
					return err
				}
				return nil
			}, 5*time.Minute, 5*time.Second).Should(BeNil())
			// check basic expectations for the lease
			Expect(expectLease(lease, nodeName)).To(BeNil())
			leaseDuration := time.Duration(*lease.Spec.LeaseDurationSeconds) * time.Second

			By("verify NodeStatus report period is longer than lease duration")
			// NodeStatus is reported from node to master when there is some change or
			// enough time has passed. So for here, keep checking the time diff
			// between 2 NodeStatus report, until it is longer than lease duration
			// (the same as nodeMonitorGracePeriod), or it doesn't change for at least leaseDuration
			lastHeartbeatTime, lastStatus := getHeartbeatTimeAndStatus(f.ClientSet, nodeName)
			lastObserved := time.Now()
			err = wait.Poll(time.Second, 5*time.Minute, func() (bool, error) {
				currentHeartbeatTime, currentStatus := getHeartbeatTimeAndStatus(f.ClientSet, nodeName)
				currentObserved := time.Now()

				if currentHeartbeatTime == lastHeartbeatTime {
					if currentObserved.Sub(lastObserved) > 2*leaseDuration {
						// heartbeat hasn't changed while watching for at least 2*leaseDuration, success!
						framework.Logf("node status heartbeat is unchanged for %s, was waiting for at least %s, success!", currentObserved.Sub(lastObserved), 2*leaseDuration)
						return true, nil
					}
					framework.Logf("node status heartbeat is unchanged for %s, waiting for %s", currentObserved.Sub(lastObserved), 2*leaseDuration)
					return false, nil
				}

				if currentHeartbeatTime.Sub(lastHeartbeatTime) >= leaseDuration {
					// heartbeat time changed, but the diff was greater than leaseDuration, success!
					framework.Logf("node status heartbeat changed in %s, was waiting for at least %s, success!", currentHeartbeatTime.Sub(lastHeartbeatTime), leaseDuration)
					return true, nil
				}

				if !apiequality.Semantic.DeepEqual(lastStatus, currentStatus) {
					// heartbeat time changed, but there were relevant changes in the status, keep waiting
					framework.Logf("node status heartbeat changed in %s (with other status changes), waiting for %s", currentHeartbeatTime.Sub(lastHeartbeatTime), leaseDuration)
					framework.Logf("%s", diff.ObjectReflectDiff(lastStatus, currentStatus))
					lastHeartbeatTime = currentHeartbeatTime
					lastObserved = currentObserved
					lastStatus = currentStatus
					return false, nil
				}

				// heartbeat time changed, with no other status changes, in less time than we expected, so fail.
				return false, fmt.Errorf("node status heartbeat changed in %s (with no other status changes), was waiting for %s", currentHeartbeatTime.Sub(lastHeartbeatTime), leaseDuration)
			})
			// a timeout is acceptable, since it means we waited 5 minutes and didn't see any unwarranted node status updates
			if err != nil && err != wait.ErrWaitTimeout {
				Expect(err).NotTo(HaveOccurred(), "error waiting for infrequent nodestatus update")
			}

			By("verify node is still in ready status even though node status report is infrequent")
			// This check on node status is only meaningful when this e2e test is
			// running as cluster e2e test, because node e2e test does not create and
			// run controller manager, i.e., no node lifecycle controller.
			node, err := f.ClientSet.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
			Expect(err).To(BeNil())
			_, readyCondition := testutils.GetNodeCondition(&node.Status, corev1.NodeReady)
			Expect(readyCondition.Status).To(Equal(corev1.ConditionTrue))
		})
	})
})

func getHeartbeatTimeAndStatus(clientSet clientset.Interface, nodeName string) (time.Time, corev1.NodeStatus) {
	node, err := clientSet.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	Expect(err).To(BeNil())
	_, readyCondition := testutils.GetNodeCondition(&node.Status, corev1.NodeReady)
	Expect(readyCondition.Status).To(Equal(corev1.ConditionTrue))
	heartbeatTime := readyCondition.LastHeartbeatTime.Time
	readyCondition.LastHeartbeatTime = metav1.Time{}
	return heartbeatTime, node.Status
}

func expectLease(lease *coordv1beta1.Lease, nodeName string) error {
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
	if *lease.Spec.HolderIdentity != nodeName {
		return fmt.Errorf("Spec.HolderIdentity (%v) should match the node name (%v)", *lease.Spec.HolderIdentity, nodeName)
	}
	return nil
}
