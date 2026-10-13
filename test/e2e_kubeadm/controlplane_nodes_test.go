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
	"context"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// Define container for all the test specification aimed at verifying
// that kubeadm configures the control-plane node as expected
var _ = Describe("control-plane node", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("control-plane node")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	// Important! please note that this test can't be run on single-node clusters
	// in case you can skip this test with SKIP=multi-node
	ginkgo.It("should be labelled and tainted [multi-node]", func(ctx context.Context) {
		// get all control-plane nodes (and this implicitly checks that node are properly labeled)
		controlPlanes := framework.GetControlPlaneNodes(ctx, f.ClientSet)

		// checks if there is at least one control-plane node
		gomega.Expect(controlPlanes.Items).NotTo(gomega.BeEmpty(), "at least one node with label %s should exist. if you are running test on a single-node cluster, you can skip this test with SKIP=multi-node", framework.ControlPlaneLabel)

		// checks that the control-plane nodes have the expected taints
		for _, cp := range controlPlanes.Items {
			e2enode.ExpectNodeHasTaint(ctx, f.ClientSet, cp.GetName(), &corev1.Taint{Key: framework.ControlPlaneLabel, Effect: corev1.TaintEffectNoSchedule})
		}
	})
})
