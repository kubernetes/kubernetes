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

package apps

import (
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// schedulingTimeout is longer specifically because sometimes we need to wait
// awhile to guarantee that we've been patient waiting for something ordinary
// to happen: a pod to get scheduled and move into Ready
const (
	bigClusterSize    = 7
	schedulingTimeout = 10 * time.Minute
	timeout           = 60 * time.Second
)

var _ = SIGDescribe("DisruptionController", func() {
	f := framework.NewDefaultFramework("disruption")
	var ns string
	var cs kubernetes.Interface

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
	})

	ginkgo.It("should create a PodDisruptionBudget", func() {
		createPDBMinAvailableOrDie(cs, ns, intstr.FromString("1%"))
	})

	ginkgo.It("should update PodDisruptionBudget status", func() {
		createPDBMinAvailableOrDie(cs, ns, intstr.FromInt(2))

		createPodsOrDie(cs, ns, 3)
		waitForPodsOrDie(cs, ns, 3)

		// Since disruptionAllowed starts out 0, if we see it ever become positive,
		// that means the controller is working.
		err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
			pdb, err := cs.PolicyV1beta1().PodDisruptionBudgets(ns).Get("foo", metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return pdb.Status.DisruptionsAllowed > 0, nil
		})
		framework.ExpectNoError(err)
	})

	evictionCases := []struct {
		description        string
		minAvailable       intstr.IntOrString
		maxUnavailable     intstr.IntOrString
		podCount           int
		replicaSetSize     int32
		shouldDeny         bool
		exclusive          bool
		skipForBigClusters bool
	}{
		{
			description:    "no PDB",
			minAvailable:   intstr.FromString(""),
			maxUnavailable: intstr.FromString(""),
			podCount:       1,
			shouldDeny:     false,
		}, {
			description:    "too few pods, absolute",
			minAvailable:   intstr.FromInt(2),
			maxUnavailable: intstr.FromString(""),
			podCount:       2,
			shouldDeny:     true,
		}, {
			description:    "enough pods, absolute",
			minAvailable:   intstr.FromInt(2),
			maxUnavailable: intstr.FromString(""),
			podCount:       3,
			shouldDeny:     false,
		}, {
			description:    "enough pods, replicaSet, percentage",
			minAvailable:   intstr.FromString("90%"),
			maxUnavailable: intstr.FromString(""),
			replicaSetSize: 10,
			exclusive:      false,
			shouldDeny:     false,
		}, {
			description:    "too few pods, replicaSet, percentage",
			minAvailable:   intstr.FromString("90%"),
			maxUnavailable: intstr.FromString(""),
			replicaSetSize: 10,
			exclusive:      true,
			shouldDeny:     true,
			// This tests assumes that there is less than replicaSetSize nodes in the cluster.
			skipForBigClusters: true,
		},
		{
			description:    "maxUnavailable allow single eviction, percentage",
			minAvailable:   intstr.FromString(""),
			maxUnavailable: intstr.FromString("10%"),
			replicaSetSize: 10,
			exclusive:      false,
			shouldDeny:     false,
		},
		{
			description:    "maxUnavailable deny evictions, integer",
			minAvailable:   intstr.FromString(""),
			maxUnavailable: intstr.FromInt(1),
			replicaSetSize: 10,
			exclusive:      true,
			shouldDeny:     true,
			// This tests assumes that there is less than replicaSetSize nodes in the cluster.
			skipForBigClusters: true,
		},
	}
	for i := range evictionCases {
		c := evictionCases[i]
		expectation := "should allow an eviction"
		if c.shouldDeny {
			expectation = "should not allow an eviction"
		}
		ginkgo.It(fmt.Sprintf("evictions: %s => %s", c.description, expectation), func() {
			if c.skipForBigClusters {
				e2eskipper.SkipUnlessNodeCountIsAtMost(bigClusterSize - 1)
			}
			createPodsOrDie(cs, ns, c.podCount)
			if c.replicaSetSize > 0 {
				createReplicaSetOrDie(cs, ns, c.replicaSetSize, c.exclusive)
			}

			if c.minAvailable.String() != "" {
				createPDBMinAvailableOrDie(cs, ns, c.minAvailable)
			}

			if c.maxUnavailable.String() != "" {
				createPDBMaxUnavailableOrDie(cs, ns, c.maxUnavailable)
			}

			// Locate a running pod.
			pod, err := locateRunningPod(cs, ns)
			framework.ExpectNoError(err)

			e := &policyv1beta1.Eviction{
				ObjectMeta: metav1.ObjectMeta{
					Name:      pod.Name,
					Namespace: ns,
				},
			}

			if c.shouldDeny {
				err = cs.CoreV1().Pods(ns).Evict(e)
				gomega.Expect(err).Should(gomega.MatchError("Cannot evict pod as it would violate the pod's disruption budget."))
			} else {
				// Only wait for running pods in the "allow" case
				// because one of shouldDeny cases relies on the
				// replicaSet not fitting on the cluster.
				waitForPodsOrDie(cs, ns, c.podCount+int(c.replicaSetSize))

				// Since disruptionAllowed starts out false, if an eviction is ever allowed,
				// that means the controller is working.
				err = wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
					err = cs.CoreV1().Pods(ns).Evict(e)
					if err != nil {
						return false, nil
					}
					return true, nil
				})
				framework.ExpectNoError(err)
			}
		})
	}

	ginkgo.It("should block an eviction until the PDB is updated to allow it", func() {
		ginkgo.By("Creating a pdb that targets all three pods in a test replica set")
		createPDBMinAvailableOrDie(cs, ns, intstr.FromInt(3))
		createReplicaSetOrDie(cs, ns, 3, false)

		ginkgo.By("First trying to evict a pod which shouldn't be evictable")
		pod, err := locateRunningPod(cs, ns)
		framework.ExpectNoError(err)

		waitForPodsOrDie(cs, ns, 3) // make sure that they are running and so would be evictable with a different pdb
		e := &policyv1beta1.Eviction{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pod.Name,
				Namespace: ns,
			},
		}
		err = cs.CoreV1().Pods(ns).Evict(e)
		gomega.Expect(err).Should(gomega.MatchError("Cannot evict pod as it would violate the pod's disruption budget."))

		ginkgo.By("Updating the pdb to allow a pod to be evicted")
		updatePDBMinAvailableOrDie(cs, ns, intstr.FromInt(2))

		ginkgo.By("Trying to evict the same pod we tried earlier which should now be evictable")
		waitForPodsOrDie(cs, ns, 3)
		waitForPdbToObserveHealthyPods(cs, ns, 3)
		err = cs.CoreV1().Pods(ns).Evict(e)
		framework.ExpectNoError(err) // the eviction is now allowed
	})
})

func createPDBMinAvailableOrDie(cs kubernetes.Interface, ns string, minAvailable intstr.IntOrString) {
	pdb := policyv1beta1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
		},
		Spec: policyv1beta1.PodDisruptionBudgetSpec{
			Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			MinAvailable: &minAvailable,
		},
	}
	_, err := cs.PolicyV1beta1().PodDisruptionBudgets(ns).Create(&pdb)
	framework.ExpectNoError(err, "Waiting for the pdb to be created with minAvailable %d in namespace %s", minAvailable.IntVal, ns)
	waitForPdbToBeProcessed(cs, ns)
}

func createPDBMaxUnavailableOrDie(cs kubernetes.Interface, ns string, maxUnavailable intstr.IntOrString) {
	pdb := policyv1beta1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: ns,
		},
		Spec: policyv1beta1.PodDisruptionBudgetSpec{
			Selector:       &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
			MaxUnavailable: &maxUnavailable,
		},
	}
	_, err := cs.PolicyV1beta1().PodDisruptionBudgets(ns).Create(&pdb)
	framework.ExpectNoError(err, "Waiting for the pdb to be created with maxUnavailable %d in namespace %s", maxUnavailable.IntVal, ns)
	waitForPdbToBeProcessed(cs, ns)
}

func updatePDBMinAvailableOrDie(cs kubernetes.Interface, ns string, minAvailable intstr.IntOrString) {
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		old, err := cs.PolicyV1beta1().PodDisruptionBudgets(ns).Get("foo", metav1.GetOptions{})
		if err != nil {
			return err
		}
		old.Spec.MinAvailable = &minAvailable
		if _, err := cs.PolicyV1beta1().PodDisruptionBudgets(ns).Update(old); err != nil {
			return err
		}
		return nil
	})

	framework.ExpectNoError(err, "Waiting for the pdb update to be processed in namespace %s", ns)
	waitForPdbToBeProcessed(cs, ns)
}

func createPodsOrDie(cs kubernetes.Interface, ns string, n int) {
	for i := 0; i < n; i++ {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("pod-%d", i),
				Namespace: ns,
				Labels:    map[string]string{"foo": "bar"},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "busybox",
						Image: imageutils.GetE2EImage(imageutils.EchoServer),
					},
				},
				RestartPolicy: v1.RestartPolicyAlways,
			},
		}

		_, err := cs.CoreV1().Pods(ns).Create(pod)
		framework.ExpectNoError(err, "Creating pod %q in namespace %q", pod.Name, ns)
	}
}

func waitForPodsOrDie(cs kubernetes.Interface, ns string, n int) {
	ginkgo.By("Waiting for all pods to be running")
	err := wait.PollImmediate(framework.Poll, schedulingTimeout, func() (bool, error) {
		pods, err := cs.CoreV1().Pods(ns).List(metav1.ListOptions{LabelSelector: "foo=bar"})
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
		for i := range pods.Items {
			pod := pods.Items[i]
			if podutil.IsPodReady(&pod) {
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

func createReplicaSetOrDie(cs kubernetes.Interface, ns string, size int32, exclusive bool) {
	container := v1.Container{
		Name:  "busybox",
		Image: imageutils.GetE2EImage(imageutils.EchoServer),
	}
	if exclusive {
		container.Ports = []v1.ContainerPort{
			{HostPort: 5555, ContainerPort: 5555},
		}
	}

	rs := &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "rs",
			Namespace: ns,
		},
		Spec: appsv1.ReplicaSetSpec{
			Replicas: &size,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"foo": "bar"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{container},
				},
			},
		},
	}

	_, err := cs.AppsV1().ReplicaSets(ns).Create(rs)
	framework.ExpectNoError(err, "Creating replica set %q in namespace %q", rs.Name, ns)
}

func locateRunningPod(cs kubernetes.Interface, ns string) (pod *v1.Pod, err error) {
	ginkgo.By("locating a running pod")
	err = wait.PollImmediate(framework.Poll, schedulingTimeout, func() (bool, error) {
		podList, err := cs.CoreV1().Pods(ns).List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}

		for i := range podList.Items {
			p := podList.Items[i]
			if podutil.IsPodReady(&p) {
				pod = &p
				return true, nil
			}
		}

		return false, nil
	})
	return pod, err
}

func waitForPdbToBeProcessed(cs kubernetes.Interface, ns string) {
	ginkgo.By("Waiting for the pdb to be processed")
	err := wait.PollImmediate(framework.Poll, schedulingTimeout, func() (bool, error) {
		pdb, err := cs.PolicyV1beta1().PodDisruptionBudgets(ns).Get("foo", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if pdb.Status.ObservedGeneration < pdb.Generation {
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Waiting for the pdb to be processed in namespace %s", ns)
}

func waitForPdbToObserveHealthyPods(cs kubernetes.Interface, ns string, healthyCount int32) {
	ginkgo.By("Waiting for the pdb to observed all healthy pods")
	err := wait.PollImmediate(framework.Poll, wait.ForeverTestTimeout, func() (bool, error) {
		pdb, err := cs.PolicyV1beta1().PodDisruptionBudgets(ns).Get("foo", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if pdb.Status.CurrentHealthy != healthyCount {
			return false, nil
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Waiting for the pdb in namespace %s to observed %d healthy pods", ns, healthyCount)
}
