/*
Copyright 2017 The Kubernetes Authors.

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

package apps

import (
	"context"
	"github.com/onsi/ginkgo/v2"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"

	"k8s.io/kubernetes/test/e2e/framework"
	e2estatefulset "k8s.io/kubernetes/test/e2e/framework/statefulset"
	"k8s.io/kubernetes/test/e2e/upgrades"
)

// createStatefulSetService creates a Headless Service with Name name and Selector set to match labels.
func createStatefulSetService(name string, labels map[string]string) *v1.Service {
	headlessService := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ServiceSpec{
			Selector: labels,
		},
	}
	headlessService.Spec.Ports = []v1.ServicePort{
		{Port: 80, Name: "http", Protocol: v1.ProtocolTCP},
	}
	headlessService.Spec.ClusterIP = "None"
	return headlessService
}

// StatefulSetUpgradeTest implements an upgrade test harness for StatefulSet upgrade testing.
type StatefulSetUpgradeTest struct {
	service *v1.Service
	set     *appsv1.StatefulSet
}

// Name returns the tracking name of the test.
func (StatefulSetUpgradeTest) Name() string { return "[sig-apps] statefulset-upgrade" }

// Skip returns true when this test can be skipped.
func (StatefulSetUpgradeTest) Skip(upgCtx upgrades.UpgradeContext) bool {
	minVersion := version.MustParseSemantic("1.5.0")

	for _, vCtx := range upgCtx.Versions {
		if vCtx.Version.LessThan(minVersion) {
			return true
		}
	}
	return false
}

// Setup creates a StatefulSet and a HeadlessService. It verifies the basic SatefulSet properties
func (t *StatefulSetUpgradeTest) Setup(f *framework.Framework) {
	ssName := "ss"
	labels := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	headlessSvcName := "test"
	statefulPodMounts := []v1.VolumeMount{{Name: "datadir", MountPath: "/data/"}}
	podMounts := []v1.VolumeMount{{Name: "home", MountPath: "/home"}}
	ns := f.Namespace.Name
	t.set = e2estatefulset.NewStatefulSet(ssName, ns, headlessSvcName, 2, statefulPodMounts, podMounts, labels)
	t.service = createStatefulSetService(ssName, labels)
	*(t.set.Spec.Replicas) = 3
	e2estatefulset.PauseNewPods(t.set)

	ginkgo.By("Creating service " + headlessSvcName + " in namespace " + ns)
	_, err := f.ClientSet.CoreV1().Services(ns).Create(context.TODO(), t.service, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Creating statefulset " + ssName + " in namespace " + ns)
	*(t.set.Spec.Replicas) = 3
	_, err = f.ClientSet.AppsV1().StatefulSets(ns).Create(context.TODO(), t.set, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Saturating stateful set " + t.set.Name)
	e2estatefulset.Saturate(f.ClientSet, t.set)
	t.verify(f)
	t.restart(f)
	t.verify(f)
}

// Test waits for the upgrade to complete and verifies the StatefulSet basic functionality
func (t *StatefulSetUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	<-done
	t.verify(f)
}

// Teardown deletes all StatefulSets
func (t *StatefulSetUpgradeTest) Teardown(f *framework.Framework) {
	e2estatefulset.DeleteAllStatefulSets(f.ClientSet, t.set.Name)
}

func (t *StatefulSetUpgradeTest) verify(f *framework.Framework) {
	ginkgo.By("Verifying statefulset mounted data directory is usable")
	framework.ExpectNoError(e2estatefulset.CheckMount(f.ClientSet, t.set, "/data"))

	ginkgo.By("Verifying statefulset provides a stable hostname for each pod")
	framework.ExpectNoError(e2estatefulset.CheckHostname(f.ClientSet, t.set))

	ginkgo.By("Verifying statefulset set proper service name")
	framework.ExpectNoError(e2estatefulset.CheckServiceName(t.set, t.set.Spec.ServiceName))

	cmd := "echo $(hostname) > /data/hostname; sync;"
	ginkgo.By("Running " + cmd + " in all stateful pods")
	framework.ExpectNoError(e2estatefulset.ExecInStatefulPods(f.ClientSet, t.set, cmd))
}

func (t *StatefulSetUpgradeTest) restart(f *framework.Framework) {
	ginkgo.By("Restarting statefulset " + t.set.Name)
	e2estatefulset.Restart(f.ClientSet, t.set)
	e2estatefulset.WaitForRunningAndReady(f.ClientSet, *t.set.Spec.Replicas, t.set)
}
