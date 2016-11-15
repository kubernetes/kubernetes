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

package e2e

import (
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/autoscaling"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	awscloud "k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
)

const (
	serveHostnameImage        = "gcr.io/google_containers/serve_hostname:v1.4"
	resizeNodeReadyTimeout    = 2 * time.Minute
	resizeNodeNotReadyTimeout = 2 * time.Minute
	nodeReadinessTimeout      = 3 * time.Minute
	podNotReadyTimeout        = 1 * time.Minute
	podReadyTimeout           = 2 * time.Minute
	testPort                  = 9376
)

func ResizeGroup(group string, size int32) error {
	if framework.TestContext.ReportDir != "" {
		framework.CoreDump(framework.TestContext.ReportDir)
		defer framework.CoreDump(framework.TestContext.ReportDir)
	}
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "resize",
			group, fmt.Sprintf("--size=%v", size),
			"--project="+framework.TestContext.CloudConfig.ProjectID, "--zone="+framework.TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			framework.Logf("Failed to resize node instance group: %v", string(output))
		}
		return err
	} else if framework.TestContext.Provider == "aws" {
		client := autoscaling.New(session.New())
		return awscloud.ResizeInstanceGroup(client, group, int(size))
	} else {
		return fmt.Errorf("Provider does not support InstanceGroups")
	}
}

func GetGroupNodes(group string) ([]string, error) {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
			"list-instances", group, "--project="+framework.TestContext.CloudConfig.ProjectID,
			"--zone="+framework.TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			return nil, err
		}
		re := regexp.MustCompile(".*RUNNING")
		lines := re.FindAllString(string(output), -1)
		for i, line := range lines {
			lines[i] = line[:strings.Index(line, " ")]
		}
		return lines, nil
	} else {
		return nil, fmt.Errorf("provider does not support InstanceGroups")
	}
}

func GroupSize(group string) (int, error) {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
			"list-instances", group, "--project="+framework.TestContext.CloudConfig.ProjectID,
			"--zone="+framework.TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			return -1, err
		}
		re := regexp.MustCompile("RUNNING")
		return len(re.FindAllString(string(output), -1)), nil
	} else if framework.TestContext.Provider == "aws" {
		client := autoscaling.New(session.New())
		instanceGroup, err := awscloud.DescribeInstanceGroup(client, group)
		if err != nil {
			return -1, fmt.Errorf("error describing instance group: %v", err)
		}
		if instanceGroup == nil {
			return -1, fmt.Errorf("instance group not found: %s", group)
		}
		return instanceGroup.CurrentSize()
	} else {
		return -1, fmt.Errorf("provider does not support InstanceGroups")
	}
}

func WaitForGroupSize(group string, size int32) error {
	timeout := 10 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		currentSize, err := GroupSize(group)
		if err != nil {
			framework.Logf("Failed to get node instance group size: %v", err)
			continue
		}
		if currentSize != int(size) {
			framework.Logf("Waiting for node instance group size %d, current size %d", size, currentSize)
			continue
		}
		framework.Logf("Node instance group has reached the desired size %d", size)
		return nil
	}
	return fmt.Errorf("timeout waiting %v for node instance group size to be %d", timeout, size)
}

func svcByName(name string, port int) *api.Service {
	return &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ServiceSpec{
			Type: api.ServiceTypeNodePort,
			Selector: map[string]string{
				"name": name,
			},
			Ports: []api.ServicePort{{
				Port:       int32(port),
				TargetPort: intstr.FromInt(port),
			}},
		},
	}
}

func newSVCByName(c clientset.Interface, ns, name string) error {
	_, err := c.Core().Services(ns).Create(svcByName(name, testPort))
	return err
}

func rcByNamePort(name string, replicas int32, image string, port int, protocol api.Protocol,
	labels map[string]string, gracePeriod *int64) *api.ReplicationController {

	return rcByNameContainer(name, replicas, image, labels, api.Container{
		Name:  name,
		Image: image,
		Ports: []api.ContainerPort{{ContainerPort: int32(port), Protocol: protocol}},
	}, gracePeriod)
}

func rcByNameContainer(name string, replicas int32, image string, labels map[string]string, c api.Container,
	gracePeriod *int64) *api.ReplicationController {

	zeroGracePeriod := int64(0)

	// Add "name": name to the labels, overwriting if it exists.
	labels["name"] = name
	if gracePeriod == nil {
		gracePeriod = &zeroGracePeriod
	}
	return &api.ReplicationController{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{
				"name": name,
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: labels,
				},
				Spec: api.PodSpec{
					Containers:                    []api.Container{c},
					TerminationGracePeriodSeconds: gracePeriod,
				},
			},
		},
	}
}

// newRCByName creates a replication controller with a selector by name of name.
func newRCByName(c clientset.Interface, ns, name string, replicas int32, gracePeriod *int64) (*api.ReplicationController, error) {
	By(fmt.Sprintf("creating replication controller %s", name))
	return c.Core().ReplicationControllers(ns).Create(rcByNamePort(
		name, replicas, serveHostnameImage, 9376, api.ProtocolTCP, map[string]string{}, gracePeriod))
}

func resizeRC(c clientset.Interface, ns, name string, replicas int32) error {
	rc, err := c.Core().ReplicationControllers(ns).Get(name)
	if err != nil {
		return err
	}
	rc.Spec.Replicas = replicas
	_, err = c.Core().ReplicationControllers(rc.Namespace).Update(rc)
	return err
}

var _ = framework.KubeDescribe("Nodes [Disruptive]", func() {
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
	framework.KubeDescribe("Resize [Slow]", func() {
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
			if err := ResizeGroup(group, int32(framework.TestContext.CloudConfig.NumNodes)); err != nil {
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
			if err := WaitForGroupSize(group, int32(framework.TestContext.CloudConfig.NumNodes)); err != nil {
				framework.Failf("Couldn't restore the original node instance group size: %v", err)
			}
			if err := framework.WaitForClusterSize(c, framework.TestContext.CloudConfig.NumNodes, 10*time.Minute); err != nil {
				framework.Failf("Couldn't restore the original cluster size: %v", err)
			}
			// Many e2e tests assume that the cluster is fully healthy before they start.  Wait until
			// the cluster is restored to health.
			By("waiting for system pods to successfully restart")
			err := framework.WaitForPodsRunningReady(c, api.NamespaceSystem, systemPodsNo, framework.PodReadyBeforeTimeout, ignoreLabels)
			Expect(err).NotTo(HaveOccurred())
			By("waiting for image prepulling pods to complete")
			framework.WaitForPodsSuccess(c, api.NamespaceSystem, framework.ImagePullerLabels, imagePrePullingTimeout)
		})

		It("should be able to delete nodes", func() {
			// Create a replication controller for a service that serves its hostname.
			// The source for the Docker container kubernetes/serve_hostname is in contrib/for-demos/serve_hostname
			name := "my-hostname-delete-node"
			replicas := int32(framework.TestContext.CloudConfig.NumNodes)
			newRCByName(c, ns, name, replicas, nil)
			err := framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("decreasing cluster size to %d", replicas-1))
			err = ResizeGroup(group, replicas-1)
			Expect(err).NotTo(HaveOccurred())
			err = WaitForGroupSize(group, replicas-1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForClusterSize(c, int(replicas-1), 10*time.Minute)
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
			newSVCByName(c, ns, name)
			replicas := int32(framework.TestContext.CloudConfig.NumNodes)
			newRCByName(c, ns, name, replicas, nil)
			err := framework.VerifyPods(c, ns, name, true, replicas)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing cluster size to %d", replicas+1))
			err = ResizeGroup(group, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = WaitForGroupSize(group, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForClusterSize(c, int(replicas+1), 10*time.Minute)
			Expect(err).NotTo(HaveOccurred())

			By(fmt.Sprintf("increasing size of the replication controller to %d and verifying all pods are running", replicas+1))
			err = resizeRC(c, ns, name, replicas+1)
			Expect(err).NotTo(HaveOccurred())
			err = framework.VerifyPods(c, ns, name, true, replicas+1)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})
