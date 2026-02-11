/*
Copyright 2025 The Kubernetes Authors.

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

package dra

import (
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// deployDevicePlugin deploys https://github.com/kubernetes/kubernetes/tree/111a2a0d2dfe13639724506f674bc4f342ccfbab/test/images/sample-device-plugin
// on the chosen nodes.
//
// A test using this must run serially because the example plugin uses the hard-coded "example.com/resource"
// extended resource name (returned as result) and deploying it twice for the same nodes from different
// tests would conflict.
func deployDevicePlugin(tCtx ktesting.TContext, f *framework.Framework, nodeNames []string, skipCleanup bool) v1.ResourceName {
	ginkgo.By("Deploy Sample Device Plugin DaemonSet")
	err := utils.CreateFromManifests(tCtx, f, f.Namespace, func(item interface{}) error {
		switch item := item.(type) {
		case *appsv1.DaemonSet:
			item.Spec.Template.Spec.Affinity = &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "kubernetes.io/hostname",
										Operator: v1.NodeSelectorOpIn,
										Values:   nodeNames,
									},
								},
							},
						},
					},
				},
			}
		}
		return nil
	}, e2enode.SampleDevicePluginDSYAML)
	framework.ExpectNoError(err, "deploy Sample Device Plugin DaemonSet")

	if !skipCleanup {
		tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
			undeployDevicePlugin(tCtx, f, nodeNames)
		})
	}

	// Wait for the Kubelet to update resource Allocatable
	// based on the device plugin resources reported by the sample device plugin.
	for _, nodeName := range nodeNames {
		gomega.Eventually(tCtx, func() int64 {
			node, err := f.ClientSet.CoreV1().Nodes().Get(tCtx, nodeName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			return e2enode.CountSampleDeviceAllocatable(node)
		}).WithTimeout(f.Timeouts.PodStart).Should(gomega.BeNumerically("==", e2enode.SampleDevsAmount), "expected %d %q to be allocatable on node %q", e2enode.SampleDevsAmount, e2enode.SampleDeviceResourceName, nodeName)
	}

	return e2enode.SampleDeviceResourceName
}

func undeployDevicePlugin(tCtx ktesting.TContext, f *framework.Framework, nodeNames []string) {
	ginkgo.By("Undeploy Sample Device Plugin DaemonSet")
	err := f.ClientSet.AppsV1().DaemonSets(f.Namespace.Name).DeleteCollection(
		tCtx,
		metav1.DeleteOptions{},
		metav1.ListOptions{LabelSelector: "k8s-app=" + e2enode.SampleDevicePluginName},
	)
	framework.ExpectNoError(err, "undeploy Sample Device Plugin DaemonSet")

	// Wait for the Kubelet to update resource Allocatable to 0 after the device plugin is undeployed.
	// This is needed to ensure that the device plugin resources are not allocatable for pods after the device plugin is undeployed.
	for _, nodeName := range nodeNames {
		gomega.Eventually(tCtx, func() int64 {
			node, err := f.ClientSet.CoreV1().Nodes().Get(tCtx, nodeName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			return e2enode.CountSampleDeviceAllocatable(node)
		}).WithTimeout(f.Timeouts.PodStart).Should(gomega.BeZero(), "expected 0 %q to be allocatable on node %q after device plugin undeploy", e2enode.SampleDeviceResourceName, nodeName)
	}
}
