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
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	DNSdefaultTimeout      = 5 * time.Minute
	DNSNamespace           = "kube-system"
	ClusterAddonLabelKey   = "k8s-app"
	KubeDNSLabelName       = "kube-dns"
	DNSAutoscalerLabelName = "kube-dns-autoscaler"
)

var _ = framework.KubeDescribe("DNS horizontal autoscaling", func() {
	f := framework.NewDefaultFramework("dns-autoscaling")
	var c clientset.Interface
	var previousParams map[string]string
	DNSParams_1 := DNSParamsLinear{map[string]string{"linear": "{\"nodesPerReplica\": 1}"}, 1.0, 0.0}
	DNSParams_2 := DNSParamsLinear{map[string]string{"linear": "{\"nodesPerReplica\": 2}"}, 2.0, 0.0}
	DNSParams_3 := DNSParamsLinear{map[string]string{"linear": "{\"nodesPerReplica\": 3, \"coresPerReplica\": 3}"}, 3.0, 3.0}

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
		c = f.ClientSet

		Expect(len(framework.GetReadySchedulableNodesOrDie(c).Items)).NotTo(BeZero())

		pcm, err := fetchDNSScalingConfigMap(c)
		Expect(err).NotTo(HaveOccurred())
		previousParams = pcm.Data

		By("Replace the dns autoscaling parameters with testing parameters")
		Expect(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(DNSParams_1.data))).NotTo(HaveOccurred())
		By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear := getExpectReplicasFuncLinear(c, &DNSParams_1)
		Expect(waitForDNSReplicasSatisfied(c, getExpectReplicasLinear, DNSdefaultTimeout)).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		By("Restoring intial dns autoscaling parameters")
		Expect(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(previousParams))).NotTo(HaveOccurred())
	})

	// This test is separated because it is slow and need to run serially
	It("[Serial] [Slow] kube-dns-autoscaler should scale kube-dns pods when cluster size changed", func() {
		originalSizes := make(map[string]int)
		sum := 0
		for _, mig := range strings.Split(framework.TestContext.CloudConfig.NodeInstanceGroup, ",") {
			size, err := GroupSize(mig)
			Expect(err).NotTo(HaveOccurred())
			By(fmt.Sprintf("Initial size of %s: %d", mig, size))
			originalSizes[mig] = size
			sum += size
		}

		By("Manually increase cluster size")
		increasedSize := 0
		increasedSizes := make(map[string]int)
		for key, val := range originalSizes {
			increasedSizes[key] = val + 1
			increasedSize += increasedSizes[key]
		}
		setMigSizes(increasedSizes)
		Expect(WaitForClusterSizeFunc(c,
			func(size int) bool { return size == increasedSize }, scaleUpTimeout)).NotTo(HaveOccurred())

		By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear := getExpectReplicasFuncLinear(c, &DNSParams_1)
		Expect(waitForDNSReplicasSatisfied(c, getExpectReplicasLinear, DNSdefaultTimeout)).NotTo(HaveOccurred())

		By("Replace the dns autoscaling parameters with another testing parameters")
		Expect(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(DNSParams_3.data))).NotTo(HaveOccurred())
		By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear = getExpectReplicasFuncLinear(c, &DNSParams_3)
		Expect(waitForDNSReplicasSatisfied(c, getExpectReplicasLinear, DNSdefaultTimeout)).NotTo(HaveOccurred())

		By("Restoring cluster size")
		setMigSizes(originalSizes)
		Expect(framework.WaitForClusterSize(c, sum, scaleDownTimeout)).NotTo(HaveOccurred())

		By("Wait for kube-dns scaled to expected number")
		Expect(waitForDNSReplicasSatisfied(c, getExpectReplicasLinear, DNSdefaultTimeout)).NotTo(HaveOccurred())
	})

	It("kube-dns-autoscaler should scale kube-dns pods in both nonfaulty and faulty scenarios", func() {

		By("--- Scenario: should scale kube-dns based on changed parameters ---")
		By("Replace the dns autoscaling parameters with another testing parameters")
		Expect(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(DNSParams_3.data))).NotTo(HaveOccurred())
		By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear := getExpectReplicasFuncLinear(c, &DNSParams_3)
		Expect(waitForDNSReplicasSatisfied(c, getExpectReplicasLinear, DNSdefaultTimeout)).NotTo(HaveOccurred())

		By("--- Scenario: should re-create scaling parameters with default value when parameters got deleted ---")
		By("Delete the ConfigMap for autoscaler")
		err := deleteDNSScalingConfigMap(c)
		Expect(err).NotTo(HaveOccurred())

		By("Wait for the ConfigMap got re-created")
		configMap, err := waitForDNSConfigMapCreated(c, DNSdefaultTimeout)
		Expect(err).NotTo(HaveOccurred())

		By("Check the new created ConfigMap got the same data as we have")
		Expect(reflect.DeepEqual(previousParams, configMap.Data)).To(Equal(true))

		By("Replace the dns autoscaling parameters with another testing parameters")
		Expect(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(DNSParams_2.data))).NotTo(HaveOccurred())
		By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear = getExpectReplicasFuncLinear(c, &DNSParams_2)
		Expect(waitForDNSReplicasSatisfied(c, getExpectReplicasLinear, DNSdefaultTimeout)).NotTo(HaveOccurred())

		By("--- Scenario: should recover after autoscaler pod got deleted ---")
		By("Delete the autoscaler pod for kube-dns")
		Expect(deleteDNSAutoscalerPod(c)).NotTo(HaveOccurred())

		By("Replace the dns autoscaling parameters with another testing parameters")
		Expect(updateDNSScalingConfigMap(c, packDNSScalingConfigMap(DNSParams_1.data))).NotTo(HaveOccurred())
		By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear = getExpectReplicasFuncLinear(c, &DNSParams_1)
		Expect(waitForDNSReplicasSatisfied(c, getExpectReplicasLinear, DNSdefaultTimeout)).NotTo(HaveOccurred())
	})
})

type DNSParamsLinear struct {
	data            map[string]string
	nodesPerReplica float64
	coresPerReplica float64
}

type getExpectReplicasFunc func(c clientset.Interface) int

func getExpectReplicasFuncLinear(c clientset.Interface, params *DNSParamsLinear) getExpectReplicasFunc {
	return func(c clientset.Interface) int {
		var replicasFromNodes float64
		var replicasFromCores float64
		nodes := framework.GetReadySchedulableNodesOrDie(c).Items
		if params.nodesPerReplica > 0 {
			replicasFromNodes = math.Ceil(float64(len(nodes)) / params.nodesPerReplica)
		}
		if params.coresPerReplica > 0 {
			replicasFromCores = math.Ceil(float64(getScheduableCores(nodes)) / params.coresPerReplica)
		}
		return int(math.Max(1.0, math.Max(replicasFromNodes, replicasFromCores)))
	}
}

func getScheduableCores(nodes []api.Node) int64 {
	var sc resource.Quantity
	for _, node := range nodes {
		if !node.Spec.Unschedulable {
			sc.Add(node.Status.Capacity[api.ResourceCPU])
		}
	}

	scInt64, scOk := sc.AsInt64()
	if !scOk {
		framework.Logf("unable to compute integer values of schedulable cores in the cluster")
		return 0
	}
	return scInt64
}

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

func waitForDNSReplicasSatisfied(c clientset.Interface, getExpected getExpectReplicasFunc, timeout time.Duration) (err error) {
	var current int
	var expected int
	framework.Logf("Waiting up to %v for kube-dns to reach expected replicas", timeout)
	condition := func() (bool, error) {
		current, err = getDNSReplicas(c)
		if err != nil {
			return false, err
		}
		expected = getExpected(c)
		if current != expected {
			framework.Logf("replicas not as expected: got %v, expected %v", current, expected)
			return false, nil
		}
		return true, nil
	}

	if err = wait.Poll(2*time.Second, timeout, condition); err != nil {
		return fmt.Errorf("err waiting for DNS replicas to satisfy %v, got %v: %v", expected, current, err)
	}
	framework.Logf("kube-dns reaches expected replicas: %v", expected)
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
