/*
Copyright 2019 The Kubernetes Authors.

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

package kubeadm

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	controlPlaneTaint = "node-role.kubernetes.io/master"
)

// Define container for all the test specification aimed at verifying
// that kubeadm configures the control-plane node as expected
var _ = Describe("control-plane node", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("control-plane node")

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	// Important! please note that this test can't be run on single-node clusters
	// in case you can skip this test with SKIP=multi-node
	ginkgo.It("should be labelled and tainted [multi-node]", func() {
		// get all control-plane nodes (and this implicitly checks that node are properly labeled)
		controlPlanes := getControlPlaneNodes(f.ClientSet)

		// checks if there is at least one control-plane node
		gomega.Expect(controlPlanes.Items).NotTo(gomega.BeEmpty(), "at least one node with label %s should exist. if you are running test on a single-node cluster, you can skip this test with SKIP=multi-node", controlPlaneTaint)

		// checks that the control-plane nodes have the expected taint
		for _, cp := range controlPlanes.Items {
			framework.ExpectNodeHasTaint(f.ClientSet, cp.GetName(), &corev1.Taint{Key: controlPlaneTaint, Effect: corev1.TaintEffectNoSchedule})
		}
	})
})

func getControlPlaneNodes(c clientset.Interface) *corev1.NodeList {
	selector := labels.Set{controlPlaneTaint: ""}.AsSelector()
	masters, err := c.CoreV1().Nodes().
		List(metav1.ListOptions{LabelSelector: selector.String()})
	framework.ExpectNoError(err, "error reading control-plane nodes")

	return masters
}
