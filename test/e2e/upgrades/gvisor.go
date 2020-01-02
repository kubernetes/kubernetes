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

package upgrades

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/replicaset"
	imageutils "k8s.io/kubernetes/test/utils/image"

	. "github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

// Update the following test config to add/remove upgrade tests:
// https://gke-internal.googlesource.com/test-infra/+/refs/heads/master/prow/gob/config/team-review.googlesource.com/gke-kubernetes-tholos/runsc/team-gke-kubernetes-tholos-runsc.yaml

// GVisorUpgradeTest tests that gvisor pod works before and after
// a cluster upgrade.
type GVisorUpgradeTest struct {
}

// Name returns the tracking name of the test.
func (GVisorUpgradeTest) Name() string { return "gvisor-upgrade [sig-node]" }

const (
	gVisorReplicaSetBeforeUpgrade  = "gvisor-replicaset-before-upgrade"
	gVisorReplicaSetAfterUpgrade   = "gvisor-replicaset-after-upgrade"
	regularReplicaSetBeforeUpgrade = "regular-replicaset-before-upgrade"
	regularReplicaSetAfterUpgrade  = "regular-replicaset-after-upgrade"
	gvisorTestContainerName        = "test-gvisor"
)

// Setup creates a gvisor replicaset and a regular replicaset.
func (t *GVisorUpgradeTest) Setup(f *framework.Framework) {
	var (
		c  = f.ClientSet
		ns = f.Namespace.Name
	)
	for _, test := range []struct {
		name   string
		gvisor bool
	}{
		{gVisorReplicaSetBeforeUpgrade, true},
		{regularReplicaSetBeforeUpgrade, false},
	} {
		By(fmt.Sprintf("Creating a replicaset %q before upgrade", test.name))
		_, err := c.AppsV1().ReplicaSets(ns).Create(context.TODO(), newTestReplicaSet(test.name, test.gvisor), metav1.CreateOptions{})
		framework.ExpectNoError(err)

		By(fmt.Sprintf("Waiting for replicaset %q to have all of its replicas ready", test.name))
		framework.ExpectNoError(replicaset.WaitForReadyReplicaSet(c, ns, test.name))

		By(fmt.Sprintf("Verify whether replicaset %q runs in gvisor", test.name))
		verifygVisor(f, test.name, test.gvisor)
	}
}

// Test waits for the upgrade to complete, and then verifies that:
// * Both replicaset created before upgrade are still running;
// * New gvisor and regular replicasets can be created.
func (t *GVisorUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	<-done
	var (
		c  = f.ClientSet
		ns = f.Namespace.Name
	)
	for _, test := range []struct {
		name   string
		gvisor bool
	}{
		{gVisorReplicaSetBeforeUpgrade, true},
		{regularReplicaSetBeforeUpgrade, false},
	} {
		By(fmt.Sprintf("Verifying the replicaset %q is still running", test.name))
		framework.ExpectNoError(replicaset.WaitForReadyReplicaSet(c, ns, test.name))

		By(fmt.Sprintf("Verify whether replicaset %q runs in gvisor", test.name))
		verifygVisor(f, test.name, test.gvisor)
	}

	for _, test := range []struct {
		name   string
		gvisor bool
	}{
		{gVisorReplicaSetAfterUpgrade, true},
		{regularReplicaSetAfterUpgrade, false},
	} {
		By(fmt.Sprintf("Creating a replicaset %q after upgrade", test.name))
		_, err := c.AppsV1().ReplicaSets(ns).Create(context.TODO(), newTestReplicaSet(test.name, test.gvisor), metav1.CreateOptions{})
		framework.ExpectNoError(err)

		By(fmt.Sprintf("Waiting for replicaset %q to have all of its replicas ready", test.name))
		framework.ExpectNoError(replicaset.WaitForReadyReplicaSet(c, ns, test.name))

		By(fmt.Sprintf("Verify whether replicaset %q runs in gvisor", test.name))
		verifygVisor(f, test.name, test.gvisor)
	}
}

// Teardown cleans up any remaining resources.
func (t *GVisorUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// newTestReplicaSet returns a test replicaset that uses or not use gvisor runtime class.
func newTestReplicaSet(name string, gvisor bool) *appsv1.ReplicaSet {
	var (
		labels                    = map[string]string{"app": name}
		replicas            int32 = 1
		runtimeClassName          = "gvisor"
		runtimeClassNamePtr *string
	)
	if gvisor {
		runtimeClassNamePtr = &runtimeClassName
	}
	return &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: appsv1.ReplicaSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					RuntimeClassName: runtimeClassNamePtr,
					Containers: []v1.Container{
						{
							Name:    gvisorTestContainerName,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sleep", "10000"},
						},
					},
				},
			},
		},
	}
}

// verifygVisor exec into the container and verify whether it's running in gvisor.
func verifygVisor(f *framework.Framework, name string, gvisor bool) {
	pods, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(context.TODO(),
		metav1.ListOptions{
			LabelSelector: "app=" + name,
		})
	framework.ExpectNoError(err)
	framework.ExpectEqual(len(pods.Items), 1)
	pod := pods.Items[0]
	output := f.ExecCommandInContainer(pod.Name, gvisorTestContainerName, "dmesg")
	if gvisor {
		gomega.Expect(output).To(gomega.ContainSubstring("Starting gVisor"))
	} else {
		gomega.Expect(output).NotTo(gomega.ContainSubstring("Starting gVisor"))
	}
}
