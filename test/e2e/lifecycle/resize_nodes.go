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

package lifecycle

import (
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const resizeNodeReadyTimeout = 2 * time.Minute

func resizeRC(c clientset.Interface, ns, name string, replicas int32) error {
	rc, err := c.Core().ReplicationControllers(ns).Get(name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	*(rc.Spec.Replicas) = replicas
	_, err = c.Core().ReplicationControllers(rc.Namespace).Update(rc)
	return err
}

var _ = SIGDescribe("Nodes [Disruptive]", func() {
	f := framework.NewDefaultFramework("resize-nodes")
	var systemPodsNo int32
	var c clientset.Interface
	var ns string
	ignoreLabels := framework.ImagePullerLabels
	var group string

	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		systemPods, err := framework.GetPodsInNamespace(c, ns, ignoreLabels)
		Expect(err).NotTo(HaveOccurred())
		systemPodsNo = int32(len(systemPods))
		if strings.Index(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") >= 0 {
			framework.Failf("Test dose not support cluster setup with more than one MIG: %s", framework.TestContext.CloudConfig.NodeInstanceGroup)
		} else {
			group = framework.TestContext.CloudConfig.NodeInstanceGroup
		}
	})

	// Slow issue #13323 (8 min)
	Describe("Resize [Slow]", func() {
		var skipped bool

		BeforeEach(func() {
			skipped = true
			framework.SkipUnlessProviderIs("gce", "gke", "aws")
			framework.SkipUnlessNodeCountIsAtLeast(2)
			skipped = false
		})

		AfterEach(func() {
			if skipped {
				return
			}

			By("restoring the original node instance group size")
			if err := framework.ResizeGroup(group, int32(framework.TestContext.CloudConfig.NumNodes)); err != nil {
				framework.Failf("Couldn't restore the original node instance group size: %v", err)
			}
			// In GKE, our current tunneling setup has the potential to hold on to a broken tunnel (from a
			// rebooted/deleted node) for up to 5 minutes before all tunnels are dropped and recreated.
			// Most tests make use of some proxy feature to verify functionality. So, if a reboot test runs
			// right before a test that tries to get logs, for example, we may get unlucky and try to use a
			// closed tunnel to a node that was recently rebooted. There's no good way to framework.Poll for proxies
			// being closed, so we sleep.
			//
			// TODO(cjcullen) reduce this sleep (#19314)
			if framework.ProviderIs("gke") {
				By("waiting 5 minutes for all dead tunnels to be dropped")
				time.Sleep(5 * time.Minute)
			}
			if err := framework.WaitForGroupSize(group, int32(framework.TestContext.CloudConfig.NumNodes)); err != nil {
				framework.Failf("Couldn't restore the original node instance group size: %v", err)
			}
			if err := framework.WaitForReadyNodes(c, framework.TestContext.CloudConfig.NumNodes, 10*time.Minute); err != nil {
				framework.Failf("Couldn't restore the original cluster size: %v", err)
			}
			// Many e2e tests assume that the cluster is fully healthy before they start.  Wait until
			// the cluster is restored to health.
			By("waiting for system pods to successfully restart")
			err := framework.WaitForPodsRunningReady(c, metav1.NamespaceSystem, systemPodsNo, 0, framework.PodReadyBeforeTimeout, ignoreLabels)
			Expect(err).NotTo(HaveOccurred())
			By("waiting for image prepulling pods to complete")
			framework.WaitForPodsSuccess(c, metav1.NamespaceSystem, framework.ImagePullerLabels, framework.ImagePrePullingTimeout)
		})

		It("should be able to delete nodes", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-delete-node"
			replicas := int32(framework.TestContext.CloudConfig.NumNodes)
			common.NewRCByName(c, ns, name, replicas, nil)
			err := framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("decreasing cluster size to %d", replicas-1))
			err = framework.ResizeGroup(group, replicas-1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForGroupSize(group, replicas-1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForReadyNodes(c, int(replicas-1), 10*time.Minute)
			Expect(err).NotTo(HaveOccurred())

			By("waiting 1 minute for the watch in the podGC to catch up, remove any pods scheduled on " +
				"the now non-existent node and the RC to recreate it")
			time.Sleep(time.Minute)

			By("verifying whether the pods from the removed node are recreated")
			err = framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())
		})

		// TODO: Bug here - testName is not correct
		It("should be able to add nodes", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-add-node"
			common.NewSVCByName(c, ns, name)
			replicas := int32(framework.TestContext.CloudConfig.NumNodes)
			common.NewRCByName(c, ns, name, replicas, nil)
			err := framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing cluster size to %d", replicas+1))
			err = framework.ResizeGroup(group, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForGroupSize(group, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForReadyNodes(c, int(replicas+1), 10*time.Minute)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing size of the replication controller to %d and verifying all pods are running", replicas+1))
			err = resizeRC(c, ns, name, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.VerifyPods(c, ns, name, true, replicas+1)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})
