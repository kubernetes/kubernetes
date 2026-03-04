/*
Copyright 2024 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("PSINodeCondition", feature.PSINodeCondition, func() {
	f := framework.NewDefaultFramework("psi-node-condition-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessFeatureGateEnabled(features.PSINodeCondition)
	})

	ginkgo.It("should emit PSI Node Conditions as False upon healthy initialization", func(ctx context.Context) {
		ginkgo.By("Getting the current node")
		nodeList, err := f.ClientSet.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "failed to list nodes")
		gomega.Expect(nodeList.Items).ToNot(gomega.BeEmpty(), "expected at least one node")
		node := nodeList.Items[0]

		ginkgo.By("Verifying that all 4 PSI node conditions exist and are False")

		expectedConditions := []v1.NodeConditionType{
			v1.NodeSystemMemoryContentionPressure,
			v1.NodeSystemDiskContentionPressure,
			v1.NodeKubepodsMemoryContentionPressure,
			v1.NodeKubepodsDiskContentionPressure,
		}

		for _, conditionType := range expectedConditions {
			// Find the condition in the node status
			var matchingCondition *v1.NodeCondition
			for i := range node.Status.Conditions {
				if node.Status.Conditions[i].Type == conditionType {
					matchingCondition = &node.Status.Conditions[i]
					break
				}
			}

			// We use a custom detailed message for better debugging if the condition is unexpectedly missing (nil).
			// If missing, it implies the Kubelet feature gate didn't correctly activate the Setters, despite the test gate being enabled.
			gomega.Expect(matchingCondition).ToNot(gomega.BeNil(), "Expected condition %q to be present in Node %q", conditionType, node.Name)

			// Assert that the condition is False. A healthy, freshly booted CI node should not be under severe swap contention.
			gomega.Expect(matchingCondition.Status).To(gomega.Equal(v1.ConditionFalse), "Expected condition %q on Node %q to be False upon normal initialization", conditionType, node.Name)
		}
	})
})
