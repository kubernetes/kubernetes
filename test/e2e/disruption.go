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
	"k8s.io/client-go/1.4/pkg/api/unversioned"
	api "k8s.io/client-go/1.4/pkg/api/v1"
	policy "k8s.io/client-go/1.4/pkg/apis/policy/v1alpha1"
	"k8s.io/client-go/1.4/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.KubeDescribe("DisruptionController", func() {
	f := framework.NewDefaultFramework("disruption")
	var ns string
	var cs *release_1_4.Clientset

	BeforeEach(func() {
		cs = f.StagingClient
		ns = f.Namespace.Name
	})

	It("should create a PodDisruptionBudget", func() {
		createPodDisruptionBudgetOrDie(cs, ns, intstr.FromString("1%"))
	})

	It("should update PodDisruptionBudget status", func() {
		createPodDisruptionBudgetOrDie(cs, ns, intstr.FromInt(2))

		createPodsOrDie(cs, ns, 2)

		// Since disruptionAllowed starts out false, if we see it ever become true,
		// that means the controller is working.
		err := wait.PollImmediate(framework.Poll, 60*time.Second, func() (bool, error) {
			pdb, err := cs.Policy().PodDisruptionBudgets(ns).Get("foo")
			if err != nil {
				return false, err
			}
			return pdb.Status.PodDisruptionAllowed, nil
		})
		Expect(err).NotTo(HaveOccurred())

	})

	It("should allow an eviction when there is no PDB", func() {
		createPodsOrDie(cs, ns, 1)

		pod, err := cs.Pods(ns).Get("pod-0")
		Expect(err).NotTo(HaveOccurred())

		e := &policy.Eviction{
			ObjectMeta: api.ObjectMeta{
				Name:      pod.Name,
				Namespace: ns,
			},
		}

		err = cs.Pods(ns).Evict(e)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should not allow an eviction when too few pods", func() {
		createPodDisruptionBudgetOrDie(cs, ns, intstr.FromInt(2))

		createPodsOrDie(cs, ns, 1)

		pod, err := cs.Pods(ns).Get("pod-0")
		Expect(err).NotTo(HaveOccurred())

		e := &policy.Eviction{
			ObjectMeta: api.ObjectMeta{
				Name:      pod.Name,
				Namespace: ns,
			},
		}

		// Since disruptionAllowed starts out false, wait at least 60s hoping that
		// this gives the controller enough time to have truly set the status.
		time.Sleep(60 * time.Second)

		err = cs.Pods(ns).Evict(e)
		Expect(err).Should(MatchError("Cannot evict pod as it would violate the pod's disruption budget."))
	})

	It("should allow an eviction when enough pods", func() {
		createPodDisruptionBudgetOrDie(cs, ns, intstr.FromInt(2))

		createPodsOrDie(cs, ns, 2)

		pod, err := cs.Pods(ns).Get("pod-0")
		Expect(err).NotTo(HaveOccurred())

		e := &policy.Eviction{
			ObjectMeta: api.ObjectMeta{
				Name:      pod.Name,
				Namespace: ns,
			},
		}

		// Since disruptionAllowed starts out false, if an eviction is ever allowed,
		// that means the controller is working.
		err = wait.PollImmediate(framework.Poll, 60*time.Second, func() (bool, error) {
			err = cs.Pods(ns).Evict(e)
			if err != nil {
				return false, nil
			} else {
				return true, nil
			}
		})
		Expect(err).NotTo(HaveOccurred())
	})
})

func createPodDisruptionBudgetOrDie(cs *release_1_4.Clientset, ns string, minAvailable intstr.IntOrString) {
	pdb := policy.PodDisruptionBudget{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
		},
		Spec: policy.PodDisruptionBudgetSpec{
			Selector:     &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			MinAvailable: minAvailable,
		},
	}
	_, err := cs.Policy().PodDisruptionBudgets(ns).Create(&pdb)
	Expect(err).NotTo(HaveOccurred())
}

func createPodsOrDie(cs *release_1_4.Clientset, ns string, n int) {
	for i := 0; i < n; i++ {
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      fmt.Sprintf("pod-%d", i),
				Namespace: ns,
				Labels:    map[string]string{"foo": "bar"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "busybox",
						Image: "gcr.io/google_containers/echoserver:1.4",
					},
				},
				RestartPolicy: api.RestartPolicyAlways,
			},
		}

		_, err := cs.Pods(ns).Create(pod)
		framework.ExpectNoError(err, "Creating pod %q in namespace %q", pod.Name, ns)
	}
}
