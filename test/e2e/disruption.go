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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	release_1_4 "k8s.io/client-go/1.4/kubernetes"
	api "k8s.io/client-go/1.4/pkg/api"
	"k8s.io/client-go/1.4/pkg/api/unversioned"
	apiv1 "k8s.io/client-go/1.4/pkg/api/v1"
	policy "k8s.io/client-go/1.4/pkg/apis/policy/v1alpha1"
	"k8s.io/client-go/1.4/pkg/labels"
	"k8s.io/client-go/1.4/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.KubeDescribe("DisruptionController [Feature:PodDisruptionbudget]", func() {
	f := framework.NewDefaultFramework("disruption")
	var ns string
	var cs *release_1_4.Clientset

	BeforeEach(func() {
		cs = f.StagingClient
		ns = f.Namespace.Name
	})

	It("should create a PodDisruptionBudget", func() {
		pdb := policy.PodDisruptionBudget{
			ObjectMeta: apiv1.ObjectMeta{
				Name:      "foo",
				Namespace: ns,
			},
			Spec: policy.PodDisruptionBudgetSpec{
				Selector:     &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				MinAvailable: intstr.FromString("1%"),
			},
		}
		_, err := cs.Policy().PodDisruptionBudgets(ns).Create(&pdb)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should update PodDisruptionBudget status", func() {
		pdb := policy.PodDisruptionBudget{
			ObjectMeta: apiv1.ObjectMeta{
				Name:      "foo",
				Namespace: ns,
			},
			Spec: policy.PodDisruptionBudgetSpec{
				Selector:     &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				MinAvailable: intstr.FromInt(2),
			},
		}
		_, err := cs.Policy().PodDisruptionBudgets(ns).Create(&pdb)
		Expect(err).NotTo(HaveOccurred())

		createPodsOrDie(cs, ns, 3)
		waitForPodsOrDie(cs, ns, 3)

		err = wait.PollImmediate(framework.Poll, 60*time.Second, func() (bool, error) {
			pdb, err := cs.Policy().PodDisruptionBudgets(ns).Get("foo")
			if err != nil {
				return false, err
			}
			return pdb.Status.PodDisruptionAllowed, nil
		})
		Expect(err).NotTo(HaveOccurred())

	})

})

func createPodsOrDie(cs *release_1_4.Clientset, ns string, n int) {
	for i := 0; i < n; i++ {
		pod := &apiv1.Pod{
			ObjectMeta: apiv1.ObjectMeta{
				Name:      fmt.Sprintf("pod-%d", i),
				Namespace: ns,
				Labels:    map[string]string{"foo": "bar"},
			},
			Spec: apiv1.PodSpec{
				Containers: []apiv1.Container{
					{
						Name:  "busybox",
						Image: "gcr.io/google_containers/echoserver:1.4",
					},
				},
				RestartPolicy: apiv1.RestartPolicyAlways,
			},
		}

		_, err := cs.Pods(ns).Create(pod)
		framework.ExpectNoError(err, "Creating pod %q in namespace %q", pod.Name, ns)
	}
}

func waitForPodsOrDie(cs *release_1_4.Clientset, ns string, n int) {
	By("Waiting for all pods to be running")
	err := wait.PollImmediate(framework.Poll, 10*time.Minute, func() (bool, error) {
		selector, err := labels.Parse("foo=bar")
		framework.ExpectNoError(err, "Waiting for pods in namespace %q to be ready", ns)
		pods, err := cs.Core().Pods(ns).List(api.ListOptions{LabelSelector: selector})
		if err != nil {
			return false, err
		}
		if pods == nil {
			return false, fmt.Errorf("pods is nil")
		}
		if len(pods.Items) < n {
			framework.Logf("pods: %v < %v", len(pods.Items), n)
			return false, nil
		}
		ready := 0
		for i := 0; i < n; i++ {
			if pods.Items[i].Status.Phase == apiv1.PodRunning {
				ready++
			}
		}
		if ready < n {
			framework.Logf("running pods: %v < %v", ready, n)
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Waiting for pods in namespace %q to be ready", ns)
}
