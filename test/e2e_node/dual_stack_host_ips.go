/*
Copyright 2022 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	netutils "k8s.io/utils/net"
)

var _ = common.SIGDescribe("Dual Stack Host IP [Feature:PodHostIPs]", func() {
	f := framework.NewDefaultFramework("dualstack")

	ginkgo.Context("when creating a Pod, it has PodHostIPs feature", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodHostIPs): false,
			}
		})
		ginkgo.It("should create pod, add host ips is empty", func() {
			e2eskipper.SkipIfFeatureGateEnabled(kubefeatures.PodHostIPs)

			podName := "pod-dualstack-host-ips"

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   podName,
					Labels: map[string]string{"test": "dualstack-host-ips"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "dualstack-host-ips",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
						},
					},
				},
			}

			ginkgo.By("submitting the pod to kubernetes")
			podClient := f.PodClient()
			p := podClient.CreateSync(pod)

			gomega.Expect(p.Status.HostIP).ShouldNot(gomega.BeEquivalentTo(""))
			gomega.Expect(p.Status.HostIPs).Should(gomega.BeNil())

			ginkgo.By("deleting the pod")
			err := podClient.Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(30))
			framework.ExpectNoError(err, "failed to delete pod")
		})
	})

	ginkgo.Context("when creating a Pod, it has no PodHostIPs feature", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FeatureGates = map[string]bool{
				string(features.PodHostIPs): true,
			}
		})
		ginkgo.It("should create pod, add ipv6 and ipv4 ip to host ips", func() {

			podName := "pod-dualstack-host-ips"

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   podName,
					Labels: map[string]string{"test": "dualstack-host-ips"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "dualstack-host-ips",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
						},
					},
				},
			}

			ginkgo.By("submitting the pod to kubernetes")
			podClient := f.PodClient()
			p := podClient.CreateSync(pod)

			gomega.Expect(p.Status.HostIP).ShouldNot(gomega.BeEquivalentTo(""))
			gomega.Expect(p.Status.HostIPs).ShouldNot(gomega.BeNil())

			// validate first ip in HostIPs is same as HostIP
			framework.ExpectEqual(p.Status.HostIP, p.Status.HostIPs[0].IP)
			if len(p.Status.HostIPs) > 1 {
				// assert 2 host ips belong to different families
				if netutils.IsIPv4String(p.Status.HostIPs[0].IP) == netutils.IsIPv4String(p.Status.HostIPs[1].IP) {
					framework.Failf("both internalIPs %s and %s belong to the same families", p.Status.HostIPs[0].IP, p.Status.HostIPs[1].IP)
				}
			}

			nodeList, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
			framework.ExpectNoError(err)
			for _, node := range nodeList.Items {
				if node.Name == p.Spec.NodeName {
					nodeIPs := []string{}
					for _, address := range node.Status.Addresses {
						if address.Type == v1.NodeInternalIP {
							nodeIPs = append(nodeIPs, address.Address)
						}
					}
					gomega.Expect(p.Status.HostIPs).Should(gomega.Equal(nodeIPs))
					break
				}
			}

			ginkgo.By("deleting the pod")
			err = podClient.Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(30))
			framework.ExpectNoError(err, "failed to delete pod")
		})
	})
})
