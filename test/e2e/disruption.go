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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/policy"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.KubeDescribe("DisruptionController [Feature:PodDisruptionbudget]", func() {
	f := framework.NewDefaultFramework("disruption")
	var ns string
	var c *client.Client

	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name
	})

	It("should create a PodDisruptionBudget", func() {
		pdb := policy.PodDisruptionBudget{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: ns,
			},
			Spec: policy.PodDisruptionBudgetSpec{
				Selector:     &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				MinAvailable: intstr.FromString("1%"),
			},
		}
		_, err := c.Policy().PodDisruptionBudgets(ns).Create(&pdb)
		Expect(err).NotTo(HaveOccurred())
	})

	It("should update PodDisruptionBudget status", func() {
		pdb := policy.PodDisruptionBudget{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: ns,
			},
			Spec: policy.PodDisruptionBudgetSpec{
				Selector:     &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				MinAvailable: intstr.FromInt(2),
			},
		}
		_, err := c.Policy().PodDisruptionBudgets(ns).Create(&pdb)
		Expect(err).NotTo(HaveOccurred())
		for i := 0; i < 2; i++ {
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

			_, err := c.Pods(ns).Create(pod)
			framework.ExpectNoError(err, "Creating pod %q in namespace %q", pod.Name, ns)
		}
		err = wait.PollImmediate(framework.Poll, 60*time.Second, func() (bool, error) {
			pdb, err := c.Policy().PodDisruptionBudgets(ns).Get("foo")
			if err != nil {
				return false, err
			}
			return pdb.Status.PodDisruptionAllowed, nil
		})
		Expect(err).NotTo(HaveOccurred())

	})

})
