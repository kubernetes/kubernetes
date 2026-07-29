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

package autoscaling

import (
	"context"
	"fmt"
	"math"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

// This test requires coredns to be installed on the cluster with autoscaling enabled.
// Compare your coredns manifest against the command below
// helm template coredns -n kube-system coredns/coredns --set k8sAppLabelOverride=kube-dns --set fullnameOverride=coredns --set autoscaler.enabled=true

// Constants used in dns-autoscaling test.
const (
	DNSdefaultTimeout    = 5 * time.Minute
	ClusterAddonLabelKey = "k8s-app"
	DNSLabelName         = "kube-dns"
)

var _ = SIGDescribe(feature.KubeDNSAutoscaler, "DNS horizontal autoscaling", func() {
	f := framework.NewDefaultFramework("dns-autoscaling")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var c clientset.Interface
	var previousParams map[string]string
	var configMapNames map[string]string
	var originDNSReplicasCount int
	var DNSParams1 DNSParamsLinear
	var DNSParams2 DNSParamsLinear
	var DNSParams3 DNSParamsLinear

	ginkgo.BeforeEach(func(ctx context.Context) {
		c = f.ClientSet

		nodes, err := e2enode.GetReadySchedulableNodes(ctx, c)
		framework.ExpectNoError(err)
		nodeCount := len(nodes.Items)

		ginkgo.By("Collecting original replicas count and DNS scaling params")

		// Check if we are running coredns or kube-dns, the only difference is the name of the autoscaling CM.
		// The test should be have identically on both dns providers
		provider, err := detectDNSProvider(ctx, c)
		if err != nil {
			e2eskipper.Skipf("Test expects DNS provider: %s", err)
		}

		originDNSReplicasCount, err = getDNSReplicas(ctx, c)
		framework.ExpectNoError(err)
		configMapNames = map[string]string{
			"kube-dns": "kube-dns-autoscaler",
			"coredns":  "coredns-autoscaler",
		}

		pcm, err := fetchDNSScalingConfigMap(ctx, c, configMapNames[provider])
		framework.Logf("original DNS scaling params: %v", pcm)
		framework.ExpectNoError(err)
		previousParams = pcm.Data

		if nodeCount <= 500 {
			DNSParams1 = DNSParamsLinear{
				nodesPerReplica: 1,
			}
			DNSParams2 = DNSParamsLinear{
				nodesPerReplica: 2,
			}
			DNSParams3 = DNSParamsLinear{
				nodesPerReplica: 3,
				coresPerReplica: 3,
			}
		} else {
			// In large clusters, avoid creating/deleting too many DNS pods,
			// it is supposed to be correctness test, not performance one.
			// The default setup is: 256 cores/replica, 16 nodes/replica.
			// With nodeCount > 500, nodes/13, nodes/14, nodes/15 and nodes/16
			// are different numbers.
			DNSParams1 = DNSParamsLinear{
				nodesPerReplica: 13,
			}
			DNSParams2 = DNSParamsLinear{
				nodesPerReplica: 14,
			}
			DNSParams3 = DNSParamsLinear{
				nodesPerReplica: 15,
				coresPerReplica: 15,
			}
		}
	})

	// This test is separated because it is slow and need to run serially.
	// Will take around 5 minutes to run on a 4 nodes cluster.
	f.It(f.WithSerial(), f.WithSlow(), "kube-dns-autoscaler should scale kube-dns pods when cluster size changed", func(ctx context.Context) {
		numNodes, err := e2enode.TotalRegistered(ctx, c)
		framework.ExpectNoError(err)

		configMapNames = map[string]string{
			"kube-dns": "kube-dns-autoscaler",
			"coredns":  "coredns-autoscaler",
		}
		provider, err := detectDNSProvider(ctx, c)
		framework.ExpectNoError(err)

		ginkgo.By("Replace the dns autoscaling parameters with testing parameters")
		err = updateDNSScalingConfigMap(ctx, c, packDNSScalingConfigMap(configMapNames[provider], packLinearParams(&DNSParams1)))
		framework.ExpectNoError(err)
		defer func() {
			ginkgo.By("Restoring initial dns autoscaling parameters")
			err = updateDNSScalingConfigMap(ctx, c, packDNSScalingConfigMap(configMapNames[provider], previousParams))
			framework.ExpectNoError(err)

			ginkgo.By("Wait for number of running and ready kube-dns pods recover")
			label := labels.SelectorFromSet(labels.Set(map[string]string{ClusterAddonLabelKey: DNSLabelName}))
			_, err := e2epod.WaitForPodsWithLabelRunningReady(ctx, c, metav1.NamespaceSystem, label, originDNSReplicasCount, DNSdefaultTimeout)
			framework.ExpectNoError(err)
		}()
		ginkgo.By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear := getExpectReplicasFuncLinear(ctx, c, &DNSParams1)
		err = waitForDNSReplicasSatisfied(ctx, c, getExpectReplicasLinear, DNSdefaultTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Manually increase cluster size")
		cleanupIncreasedSizeFunc := increaseClusterSize(ctx, f, c, numNodes+1)
		err = WaitForClusterSizeFunc(ctx, c,
			func(size int) bool { return size == numNodes+1 }, scaleUpTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear = getExpectReplicasFuncLinear(ctx, c, &DNSParams1)
		err = waitForDNSReplicasSatisfied(ctx, c, getExpectReplicasLinear, DNSdefaultTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Replace the dns autoscaling parameters with another testing parameters")
		err = updateDNSScalingConfigMap(ctx, c, packDNSScalingConfigMap(configMapNames[provider], packLinearParams(&DNSParams3)))
		framework.ExpectNoError(err)

		ginkgo.By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear = getExpectReplicasFuncLinear(ctx, c, &DNSParams3)
		err = waitForDNSReplicasSatisfied(ctx, c, getExpectReplicasLinear, DNSdefaultTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Restoring cluster size")
		framework.ExpectNoError(cleanupIncreasedSizeFunc())

		ginkgo.By("Wait for kube-dns scaled to expected number")
		err = waitForDNSReplicasSatisfied(ctx, c, getExpectReplicasLinear, DNSdefaultTimeout)
		framework.ExpectNoError(err)
	})

	ginkgo.It("kube-dns-autoscaler should scale kube-dns pods in both nonfaulty and faulty scenarios", func(ctx context.Context) {

		configMapNames = map[string]string{
			"kube-dns": "kube-dns-autoscaler",
			"coredns":  "coredns-autoscaler",
		}
		provider, err := detectDNSProvider(ctx, c)
		framework.ExpectNoError(err)

		ginkgo.By("Replace the dns autoscaling parameters with testing parameters")
		cm := packDNSScalingConfigMap(configMapNames[provider], packLinearParams(&DNSParams1))
		framework.Logf("Updating the following cm: %v", cm)
		err = updateDNSScalingConfigMap(ctx, c, cm)
		framework.ExpectNoError(err)
		defer func() {
			ginkgo.By("Restoring initial dns autoscaling parameters")
			err = updateDNSScalingConfigMap(ctx, c, packDNSScalingConfigMap(configMapNames[provider], previousParams))
			framework.ExpectNoError(err)
		}()
		ginkgo.By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear := getExpectReplicasFuncLinear(ctx, c, &DNSParams1)
		err = waitForDNSReplicasSatisfied(ctx, c, getExpectReplicasLinear, DNSdefaultTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("--- Scenario: should scale kube-dns based on changed parameters ---")
		ginkgo.By("Replace the dns autoscaling parameters with another testing parameters")
		err = updateDNSScalingConfigMap(ctx, c, packDNSScalingConfigMap(configMapNames[provider], packLinearParams(&DNSParams3)))
		framework.ExpectNoError(err)
		ginkgo.By("Wait for kube-dns scaled to expected number")
		getExpectReplicasLinear = getExpectReplicasFuncLinear(ctx, c, &DNSParams3)
		err = waitForDNSReplicasSatisfied(ctx, c, getExpectReplicasLinear, DNSdefaultTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("--- Scenario: should re-create scaling parameters with default value when parameters got deleted ---")
		ginkgo.By("Delete the ConfigMap for autoscaler")
		err = deleteDNSScalingConfigMap(ctx, c, configMapNames[provider])
		framework.ExpectNoError(err)

		ginkgo.By("Wait for the ConfigMap got re-created")
		_, err = waitForDNSConfigMapCreated(ctx, c, DNSdefaultTimeout, configMapNames[provider])
		framework.ExpectNoError(err)

		ginkgo.By("Replace the dns autoscaling parameters with another testing parameters")
		err = updateDNSScalingConfigMap(ctx, c, packDNSScalingConfigMap(configMapNames[provider], packLinearParams(&DNSParams2)))
		framework.ExpectNoError(err)
		ginkgo.By("Wait for kube-dns/coredns scaled to expected number")
		getExpectReplicasLinear = getExpectReplicasFuncLinear(ctx, c, &DNSParams2)
		err = waitForDNSReplicasSatisfied(ctx, c, getExpectReplicasLinear, DNSdefaultTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("--- Scenario: should recover after autoscaler pod got deleted ---")
		ginkgo.By("Delete the autoscaler pod for kube-dns/coredns")
		err = deleteDNSAutoscalerPod(ctx, c)
		framework.ExpectNoError(err)

		ginkgo.By("Replace the dns autoscaling parameters with another testing parameters")
		err = updateDNSScalingConfigMap(ctx, c, packDNSScalingConfigMap(configMapNames[provider], packLinearParams(&DNSParams1)))
		framework.ExpectNoError(err)
		ginkgo.By("Wait for kube-dns/coredns scaled to expected number")
		getExpectReplicasLinear = getExpectReplicasFuncLinear(ctx, c, &DNSParams1)
		err = waitForDNSReplicasSatisfied(ctx, c, getExpectReplicasLinear, DNSdefaultTimeout)
		framework.ExpectNoError(err)
	})
})

// DNSParamsLinear is a struct for number of DNS pods.
type DNSParamsLinear struct {
	nodesPerReplica float64
	coresPerReplica float64
	min             int
	max             int
}

type getExpectReplicasFunc func(c clientset.Interface) int

func getExpectReplicasFuncLinear(ctx context.Context, c clientset.Interface, params *DNSParamsLinear) getExpectReplicasFunc {
	return func(c clientset.Interface) int {
		var replicasFromNodes float64
		var replicasFromCores float64
		nodes, err := e2enode.GetReadyNodesIncludingTainted(ctx, c)
		framework.ExpectNoError(err)
		if params.nodesPerReplica > 0 {
			replicasFromNodes = math.Ceil(float64(len(nodes.Items)) / params.nodesPerReplica)
		}
		if params.coresPerReplica > 0 {
			replicasFromCores = math.Ceil(float64(getSchedulableCores(nodes.Items)) / params.coresPerReplica)
		}
		return int(math.Max(1.0, math.Max(replicasFromNodes, replicasFromCores)))
	}
}

func getSchedulableCores(nodes []v1.Node) int64 {
	var sc resource.Quantity
	for _, node := range nodes {
		if !node.Spec.Unschedulable {
			sc.Add(node.Status.Allocatable[v1.ResourceCPU])
		}
	}
	return sc.Value()
}

func detectDNSProvider(ctx context.Context, c clientset.Interface) (string, error) {
	cm, err := c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(ctx, "coredns-autoscaler", metav1.GetOptions{})
	if cm != nil && err == nil {
		return "coredns", nil
	}

	cm, err = c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(ctx, "kube-dns-autoscaler", metav1.GetOptions{})
	if cm != nil && err == nil {
		return "kube-dns", nil
	}

	return "", fmt.Errorf("the cluster doesn't have kube-dns or coredns autoscaling configured")
}

func fetchDNSScalingConfigMap(ctx context.Context, c clientset.Interface, configMapName string) (*v1.ConfigMap, error) {
	cm, err := c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(ctx, configMapName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	return cm, nil
}

func deleteDNSScalingConfigMap(ctx context.Context, c clientset.Interface, configMapName string) error {
	if err := c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Delete(ctx, configMapName, metav1.DeleteOptions{}); err != nil {
		return err
	}
	framework.Logf("DNS autoscaling ConfigMap deleted.")
	return nil
}

func packLinearParams(params *DNSParamsLinear) map[string]string {
	paramsMap := make(map[string]string)
	paramsMap["linear"] = fmt.Sprintf("{\"nodesPerReplica\": %v,\"coresPerReplica\": %v,\"min\": %v,\"max\": %v}",
		params.nodesPerReplica,
		params.coresPerReplica,
		params.min,
		params.max)
	return paramsMap
}

func packDNSScalingConfigMap(configMapName string, params map[string]string) *v1.ConfigMap {
	configMap := v1.ConfigMap{}
	configMap.ObjectMeta.Name = configMapName
	configMap.ObjectMeta.Namespace = metav1.NamespaceSystem
	configMap.Data = params
	return &configMap
}

func updateDNSScalingConfigMap(ctx context.Context, c clientset.Interface, configMap *v1.ConfigMap) error {
	_, err := c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Update(ctx, configMap, metav1.UpdateOptions{})
	if err != nil {
		return err
	}
	framework.Logf("DNS autoscaling ConfigMap updated.")
	return nil
}

func getDNSReplicas(ctx context.Context, c clientset.Interface) (int, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{ClusterAddonLabelKey: DNSLabelName}))
	listOpts := metav1.ListOptions{LabelSelector: label.String()}
	deployments, err := c.AppsV1().Deployments(metav1.NamespaceSystem).List(ctx, listOpts)
	if err != nil {
		return 0, err
	}
	if len(deployments.Items) != 1 {
		return 0, fmt.Errorf("expected 1 DNS deployment, got %v", len(deployments.Items))
	}

	deployment := deployments.Items[0]
	return int(*(deployment.Spec.Replicas)), nil
}

func deleteDNSAutoscalerPod(ctx context.Context, c clientset.Interface) error {
	selector, _ := labels.Parse(fmt.Sprintf("%s in (kube-dns-autoscaler, coredns-autoscaler)", ClusterAddonLabelKey))
	listOpts := metav1.ListOptions{LabelSelector: selector.String()}
	pods, err := c.CoreV1().Pods(metav1.NamespaceSystem).List(ctx, listOpts)
	if err != nil {
		return err
	}
	if len(pods.Items) != 1 {
		return fmt.Errorf("expected 1 autoscaler pod, got %v", len(pods.Items))
	}

	podName := pods.Items[0].Name
	if err := c.CoreV1().Pods(metav1.NamespaceSystem).Delete(ctx, podName, metav1.DeleteOptions{}); err != nil {
		return err
	}
	framework.Logf("DNS autoscaling pod %v deleted.", podName)
	return nil
}

func waitForDNSReplicasSatisfied(ctx context.Context, c clientset.Interface, getExpected getExpectReplicasFunc, timeout time.Duration) (err error) {
	var current int
	var expected int
	framework.Logf("Waiting up to %v for kube-dns to reach expected replicas", timeout)
	condition := func(ctx context.Context) (bool, error) {
		current, err = getDNSReplicas(ctx, c)
		if err != nil {
			return false, err
		}
		expected = getExpected(c)
		if current != expected {
			framework.Logf("Replicas not as expected: got %v, expected %v", current, expected)
			return false, nil
		}
		return true, nil
	}

	if err = wait.PollUntilContextTimeout(ctx, 2*time.Second, timeout, false, condition); err != nil {
		return fmt.Errorf("err waiting for DNS replicas to satisfy %v, got %v: %w", expected, current, err)
	}
	framework.Logf("kube-dns reaches expected replicas: %v", expected)
	return nil
}

func waitForDNSConfigMapCreated(ctx context.Context, c clientset.Interface, timeout time.Duration, configMapName string) (configMap *v1.ConfigMap, err error) {
	framework.Logf("Waiting up to %v for DNS autoscaling ConfigMap to be re-created", timeout)
	condition := func(ctx context.Context) (bool, error) {
		configMap, err = fetchDNSScalingConfigMap(ctx, c, configMapName)
		if err != nil {
			return false, nil
		}
		return true, nil
	}

	if err = wait.PollUntilContextTimeout(ctx, time.Second, timeout, false, condition); err != nil {
		return nil, fmt.Errorf("err waiting for DNS autoscaling ConfigMap got re-created: %w", err)
	}
	return configMap, nil
}
