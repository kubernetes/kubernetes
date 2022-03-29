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

package gcp

import (
	"context"
	"fmt"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

func resizeRC(c clientset.Interface, ns, name string, replicas int32) error {
	rc, err := c.CoreV1().ReplicationControllers(ns).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	*(rc.Spec.Replicas) = replicas
	_, err = c.CoreV1().ReplicationControllers(rc.Namespace).Update(context.TODO(), rc, metav1.UpdateOptions{})
	return err
}

var _ = SIGDescribe("Nodes [Disruptive]", func() {
	f := framework.NewDefaultFramework("resize-nodes")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var systemPodsNo int32
	var c clientset.Interface
	var ns string
	var group string

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
		systemPods, err := e2epod.GetPodsInNamespace(c, ns, map[string]string{})
		framework.ExpectNoError(err)
		systemPodsNo = int32(len(systemPods))
		if strings.Contains(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
			framework.Failf("Test dose not support cluster setup with more than one MIG: %s", framework.TestContext.CloudConfig.NodeInstanceGroup)
		} else {
			group = framework.TestContext.CloudConfig.NodeInstanceGroup
		}
	})

	// Slow issue #13323 (8 min)
	ginkgo.Describe("Resize [Slow]", func() {
		var originalNodeCount int32
		var skipped bool

		ginkgo.BeforeEach(func() {
			skipped = true
			e2eskipper.SkipUnlessProviderIs("gce", "gke", "aws")
			e2eskipper.SkipUnlessNodeCountIsAtLeast(2)
			skipped = false
		})

		ginkgo.AfterEach(func() {
			if skipped {
				return
			}

			ginkgo.By("restoring the original node instance group size")
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
				ginkgo.By("waiting 5 minutes for all dead tunnels to be dropped")
				time.Sleep(5 * time.Minute)
			}
			if err := framework.WaitForGroupSize(group, int32(framework.TestContext.CloudConfig.NumNodes)); err != nil {
				framework.Failf("Couldn't restore the original node instance group size: %v", err)
			}

			if err := e2enode.WaitForReadyNodes(c, int(originalNodeCount), 10*time.Minute); err != nil {
				framework.Failf("Couldn't restore the original cluster size: %v", err)
			}
			// Many e2e tests assume that the cluster is fully healthy before they start.  Wait until
			// the cluster is restored to health.
			ginkgo.By("waiting for system pods to successfully restart")
			err := e2epod.WaitForPodsRunningReady(c, metav1.NamespaceSystem, systemPodsNo, 0, framework.PodReadyBeforeTimeout, map[string]string{})
			framework.ExpectNoError(err)
		})

		ginkgo.It("should be able to delete nodes", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-delete-node"
			numNodes, err := e2enode.TotalRegistered(c)
			framework.ExpectNoError(err)
			originalNodeCount = int32(numNodes)
			common.NewRCByName(c, ns, name, originalNodeCount, nil, nil)
			err = e2epod.VerifyPods(c, ns, name, true, originalNodeCount)
			framework.ExpectNoError(err)

			targetNumNodes := int32(framework.TestContext.CloudConfig.NumNodes - 1)
			ginkgo.By(fmt.Sprintf("decreasing cluster size to %d", targetNumNodes))
			err = framework.ResizeGroup(group, targetNumNodes)
			framework.ExpectNoError(err)
			err = framework.WaitForGroupSize(group, targetNumNodes)
			framework.ExpectNoError(err)
			err = e2enode.WaitForReadyNodes(c, int(originalNodeCount-1), 10*time.Minute)
			framework.ExpectNoError(err)

			ginkgo.By("waiting 2 minutes for the watch in the podGC to catch up, remove any pods scheduled on " +
				"the now non-existent node and the RC to recreate it")
			time.Sleep(framework.NewTimeoutContextWithDefaults().PodStartShort)

			ginkgo.By("verifying whether the pods from the removed node are recreated")
			err = e2epod.VerifyPods(c, ns, name, true, originalNodeCount)
			framework.ExpectNoError(err)
		})

		// TODO: Bug here - testName is not correct
		ginkgo.It("should be able to add nodes", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-add-node"
			common.NewSVCByName(c, ns, name)
			numNodes, err := e2enode.TotalRegistered(c)
			framework.ExpectNoError(err)
			originalNodeCount = int32(numNodes)
			common.NewRCByName(c, ns, name, originalNodeCount, nil, nil)
			err = e2epod.VerifyPods(c, ns, name, true, originalNodeCount)
			framework.ExpectNoError(err)

			targetNumNodes := int32(framework.TestContext.CloudConfig.NumNodes + 1)
			ginkgo.By(fmt.Sprintf("increasing cluster size to %d", targetNumNodes))
			err = framework.ResizeGroup(group, targetNumNodes)
			framework.ExpectNoError(err)
			err = framework.WaitForGroupSize(group, targetNumNodes)
			framework.ExpectNoError(err)
			err = e2enode.WaitForReadyNodes(c, int(originalNodeCount+1), 10*time.Minute)
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("increasing size of the replication controller to %d and verifying all pods are running", originalNodeCount+1))
			err = resizeRC(c, ns, name, originalNodeCount+1)
			framework.ExpectNoError(err)
			err = e2epod.VerifyPods(c, ns, name, true, originalNodeCount+1)
			framework.ExpectNoError(err)
		})
	})
})
