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
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Eviction based on taints [Slow] [Destructive]", func() {
	f := framework.NewDefaultFramework("pod-eviction")

	framework.KubeDescribe("Pods", func() {

		It("that don't tolerate NoExecute taint should be evicted", func() {

			testTaint := api.Taint{Key: "test", Value: "test", Effect: api.TaintEffectNoExecute}
			testToleration := api.Toleration{Key: "test", Operator: api.TolerationOpEqual, Value: "test", Effect: api.TaintEffectNoExecute}

			By("updating kube-system pods with tolerations")
			systemPods, err := f.Client.Pods("kube-system").List(api.ListOptions{})
			for _, pod := range systemPods.Items {
				api.AddTolerations(&pod, testToleration)
				f.Client.Pods("kube-system").Update(&pod)
			}
			nodes := framework.GetReadySchedulableNodesOrDie(f.Client)
			node := nodes.Items[0]
			By("creating three pods in namespace " + f.Namespace.Name + ", 2 without toleration to NoExecute taint on node " + node.Name)
			pod1 := createPod("eviction-pod1-no-tolerance", node.Name)
			pod2 := createPod("eviction-pod2-no-tolerance", node.Name)
			podWithTolerance := createPod("eviction-pod3-with-tolerance", node.Name)
			api.AddTolerations(podWithTolerance, testToleration)
			pods := []*api.Pod{pod1, pod2, podWithTolerance}

			defer func() {
				By("cleaning pods in namespace " + f.Namespace.Name)
				for _, pod := range pods {
					f.PodClient().Delete(pod.Name, api.NewDeleteOptions(0))
				}
				By("removing annotations from kube-system pods")
				systemPods, err := f.Client.Pods("kube-system").List(api.ListOptions{})
				for _, pod := range systemPods.Items {
					api.RemoveTolerations(&pod, testToleration)
					_, err := f.Client.Pods("kube-system").Update(&pod)
					Expect(err).NotTo(HaveOccurred())
				}
				By("removing annotations from a chosen node")
				node, _ := f.Client.Nodes().Get(node.Name)
				api.RemoveTaints(node, testTaint)
				_, err = f.Client.Nodes().Update(node)
				Expect(err).NotTo(HaveOccurred())
			}()

			for _, pod := range pods {
				f.PodClient().Create(pod)
			}

			By("updating node " + node.Name + " with NoExecute taint")
			api.AddTaints(&node, testTaint)
			_, err = f.Client.Nodes().Update(&node)
			Expect(err).NotTo(HaveOccurred())

			By("waiting until pods without tolerations will be evicted")
			Eventually(func() error {
				pods, err := f.PodClient().List(api.ListOptions{})
				Expect(err).NotTo(HaveOccurred())
				for _, pod := range pods.Items {
					tolerations, _ := api.GetTolerationsFromPodAnnotations(pod.Annotations)
					if len(tolerations) == 0 {
						return fmt.Errorf("pod %v doesn't have required toleration and should be evicted", pod.Name)
					}
				}
				return nil

			}, 2*time.Minute, 2*time.Second).Should(BeNil())

			By("verifying that the last one is running")
			pod, err := f.PodClient().Get(podWithTolerance.Name)
			Expect(err).NotTo(HaveOccurred())
			if pod.Status.Phase != api.PodRunning {
				framework.Failf("pod %v expected to run (%v) in namespace %v", pod.Name, pod.Status.Phase, pod.Namespace)
			}
		})

		It("that are on a node under maintenance should not be evicted", func() {
			node := framework.GetReadySchedulableNodesOrDie(f.Client).Items[0]

			By("creating a pod on a node " + node.Name)
			pod1 := createPod("test1", node.Name)
			pod2 := createPod("test2", node.Name)

			pods := []*api.Pod{pod1, pod2}
			for _, pod := range pods {
				f.PodClient().Create(pod)
			}

			By("switching node " + node.Name + " into maintenance mode")
			framework.RunKubectlOrDie("maintenance", node.Name, "on")

			By("verifying that both pods are in running state on a node " + node.Name)
			for _, pod := range pods {
				if err := f.WaitForPodRunning(pod.Name); err != nil {
					framework.Failf("Pod %v is not running: %v", pod.Name, err)
				}
			}

			By("confirming that both pods are running")
			Eventually(func() error {
				pods, err := f.PodClient().List(api.ListOptions{})
				Expect(err).NotTo(HaveOccurred())

				for _, pod := range pods.Items {
					if pod.Status.Phase != api.PodRunning {
						return fmt.Errorf("pod %v expected to be running but %v", pod.Name, pod.Status.Phase)
					}
				}
				return nil

			}, 1*time.Minute, 5*time.Second).Should(BeNil())

			By("removing node " + node.Name + " from maintenance mode")
			framework.RunKubectlOrDie("maintenance", node.Name, "off")

		})

	})
})

func createPod(name, nodeName string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "nginx",
					Image: "gcr.io/google_containers/nginx-slim:0.7",
				},
			},
			NodeName: nodeName,
		},
	}
}
