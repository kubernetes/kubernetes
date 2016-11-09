/*
Copyright 2016 The Kubernetes Authors.

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
	"math"
	"reflect"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	DNSdefaultTimeout      = 5 * time.Minute
	DNSscaleUpTimeout      = 5 * time.Minute
	DNSscaleDownTimeout    = 10 * time.Minute
	DNSNamespace           = "kube-system"
	ClusterAddonLabelKey   = "k8s-app"
	KubeDNSLabelName       = "kube-dns"
	DNSAutoscalerLabelName = "kube-dns-autoscaler"
)

var _ = framework.KubeDescribe("DNS horizontal autoscaling", func() {
	f := framework.NewDefaultFramework("dns-autoscaling")
	var c clientset.Interface
	var nodeCount int
	var previousParams map[string]string
	DNSParams_1 := map[string]string{"linear": "{\"nodesPerReplica\": 1}"}
	DNSParams_2 := map[string]string{"linear": "{\"nodesPerReplica\": 2}"}
	DNSParams_3 := map[string]string{"linear": "{\"nodesPerReplica\": 3}"}

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
		c = f.ClientSet

		nodes := framework.GetReadySchedulableNodesOrDie(c)
		nodeCount = len(nodes.Items)
		Expect(nodeCount).NotTo(BeZero())

		pcm, err := fetchDNSScalingConfigMap(c)
		ExpectNoError(err)
		previousParams = pcm.Data

		By("Replace the dns autoscaling parameters with testing parameters")
		ExpectNoError(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(DNSParams_1)))
		By("Wait for kube-dns scaled to expected number")
		ExpectNoError(waitForDNSReplicasSatisfied(c, nodeCount, DNSdefaultTimeout))
	})

	AfterEach(func() {
		By("Restoring intial dns autoscaling parameters")
		ExpectNoError(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(previousParams)))
	})

	// This test is separated because it is slow and need to run serially
	It("[Serial] [Slow] kube-dns-autoscaler should scale kube-dns pods when cluster size changed", func() {
		originalSizes := make(map[string]int)
		sum := 0
		for _, mig := range strings.Split(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
			size, err := GroupSize(mig)
			ExpectNoError(err)
			By(fmt.Sprintf("Initial size of %s: %d", mig, size))
			originalSizes[mig] = size
			sum += size
		}
		Expect(nodeCount).Should(Equal(sum))

		By("Manually increase cluster size")
		increasedSize := 0
		increasedSizes := make(map[string]int)
		for key, val := range originalSizes {
			increasedSizes[key] = val + 1
			increasedSize += increasedSizes[key]
		}
		setMigSizes(increasedSizes)
		ExpectNoError(WaitForClusterSizeFunc(c,
			func(size int) bool { return size == increasedSize }, DNSscaleUpTimeout))
		By("Wait for kube-dns scaled to expected number")
		ExpectNoError(waitForDNSReplicasSatisfied(c, increasedSize, DNSdefaultTimeout))

		By("Restoring cluster size")
		setMigSizes(originalSizes)
		framework.ExpectNoError(framework.WaitForClusterSize(c, nodeCount, DNSscaleDownTimeout))
		By("Wait for kube-dns scaled to expected number")
		ExpectNoError(waitForDNSReplicasSatisfied(c, nodeCount, DNSdefaultTimeout))
	})

	It("kube-dns-autoscaler should scale kube-dns pods in both nonfaulty and faulty scenarios", func() {

		By("--- Scenario: should scale kube-dns based on changed parameters ---")
		By("Replace the dns autoscaling parameters with the second testing parameters")
		ExpectNoError(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(DNSParams_2)))
		By("Wait for kube-dns scaled to expected number")
		ExpectNoError(waitForDNSReplicasSatisfied(c, int(math.Ceil(float64(nodeCount)/2.0)), DNSdefaultTimeout))

		By("--- Scenario: should re-create scaling parameters with default value when parameters got deleted ---")
		By("Delete the ConfigMap for autoscaler")
		err := deleteDNSScalingConfigMap(c)
		ExpectNoError(err)

		By("Wait for the ConfigMap got re-created")
		configMap, err := waitForDNSConfigMapCreated(c, DNSdefaultTimeout)
		ExpectNoError(err)

		By("Check the new created ConfigMap got the same data as we have")
		Expect(reflect.DeepEqual(previousParams, configMap.Data)).To(Equal(true))

		By("Replace the dns autoscaling parameters with the second testing parameters")
		ExpectNoError(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(DNSParams_2)))
		By("Wait for kube-dns scaled to expected number")
		ExpectNoError(waitForDNSReplicasSatisfied(c, int(math.Ceil(float64(nodeCount)/2.0)), DNSdefaultTimeout))

		By("--- Scenario: should recover after autoscaler pod got deleted ---")
		By("Delete the autoscaler pod for kube-dns")
		ExpectNoError(deleteDNSAutoscalerPod(c))

		By("Replace the dns autoscaling parameters with the third testing parameters")
		ExpectNoError(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(DNSParams_3)))
		By("Wait for kube-dns scaled to expected number")
		ExpectNoError(waitForDNSReplicasSatisfied(c, int(math.Ceil(float64(nodeCount)/3.0)), DNSdefaultTimeout))
	})
})

func fetchDNSScalingConfigMap(c clientset.Interface) (*api.ConfigMap, error) {
	cm, err := c.Core().ConfigMaps(DNSNamespace).Get(DNSAutoscalerLabelName)
	if err != nil {
		return nil, err
	}
	return cm, nil
}

func deleteDNSScalingConfigMap(c clientset.Interface) error {
	if err := c.Core().ConfigMaps(DNSNamespace).Delete(DNSAutoscalerLabelName, nil); err != nil {
		return err
	}
	framework.Logf("DNS autoscaling ConfigMap deleted.")
	return nil
}

func packDNSScalingConfigMap(params map[string]string) *api.ConfigMap {
	configMap := api.ConfigMap{}
	configMap.ObjectMeta.Name = DNSAutoscalerLabelName
	configMap.ObjectMeta.Namespace = DNSNamespace
	configMap.Data = params
	return &configMap
}

func updateDNSScalingConfigMap(c clientset.Interface, configMap *api.ConfigMap) error {
	_, err := c.Core().ConfigMaps(DNSNamespace).Update(configMap)
	if err != nil {
		return err
	}
	framework.Logf("DNS autoscaling ConfigMap updated.")
	return nil
}

func getDNSReplicas(c clientset.Interface) (int, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{ClusterAddonLabelKey: KubeDNSLabelName}))
	listOpts := api.ListOptions{LabelSelector: label}
	deployments, err := c.Extensions().Deployments(DNSNamespace).List(listOpts)
	if err != nil {
		return 0, err
	}
	Expect(len(deployments.Items)).Should(Equal(1))

	deployment := deployments.Items[0]
	return int(deployment.Spec.Replicas), nil
}

func deleteDNSAutoscalerPod(c clientset.Interface) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{ClusterAddonLabelKey: DNSAutoscalerLabelName}))
	listOpts := api.ListOptions{LabelSelector: label}
	pods, err := c.Core().Pods(DNSNamespace).List(listOpts)
	if err != nil {
		return err
	}
	Expect(len(pods.Items)).Should(Equal(1))

	podName := pods.Items[0].Name
	if err := c.Core().Pods(DNSNamespace).Delete(podName, nil); err != nil {
		return err
	}
	framework.Logf("DNS autoscaling pod %v deleted.", podName)
	return nil
}

func waitForDNSReplicasSatisfied(c clientset.Interface, expected int, timeout time.Duration) (err error) {
	var current int
	framework.Logf("Waiting up to %v for kube-dns reach %v replicas", timeout, expected)
	condition := func() (bool, error) {
		current, err = getDNSReplicas(c)
		if err != nil {
			return false, err
		}
		if current != expected {
			return false, nil
		}
		return true, nil
	}

	if err = wait.Poll(time.Second, timeout, condition); err != nil {
		return fmt.Errorf("err waiting for DNS replicas to satisfy %v, got %v: %v", expected, current, err)
	}
	return nil
}

func waitForDNSConfigMapCreated(c clientset.Interface, timeout time.Duration) (configMap *api.ConfigMap, err error) {
	framework.Logf("Waiting up to %v for DNS autoscaling ConfigMap got re-created", timeout)
	condition := func() (bool, error) {
		configMap, err = fetchDNSScalingConfigMap(c)
		if err != nil {
			return false, nil
		}
		return true, nil
	}

	if err = wait.Poll(time.Second, timeout, condition); err != nil {
		return nil, fmt.Errorf("err waiting for DNS autoscaling ConfigMap got re-created: %v", err)
	}
	return configMap, nil
}
