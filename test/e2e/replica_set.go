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

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func newRS(rsName string, replicas int32, rsPodLabels map[string]string, imageName string, image string) *extensions.ReplicaSet {
	zero := int64(0)
	return &extensions.ReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Name: rsName,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: rsPodLabels,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []v1.Container{
						{
							Name:  imageName,
							Image: image,
						},
					},
				},
			},
		},
	}
}

func newPodQuota(name, number string) *v1.ResourceQuota {
	return &v1.ResourceQuota{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourcePods: resource.MustParse(number),
			},
		},
	}
}

var _ = framework.KubeDescribe("ReplicaSet", func() {
	f := framework.NewDefaultFramework("replicaset")

	It("should serve a basic image on each replica with a public image [Conformance]", func() {
		ReplicaSetServeImageOrFail(f, "basic", "gcr.io/google_containers/serve_hostname:v1.4")
	})

	It("should serve a basic image on each replica with a private image", func() {
		// requires private images
		framework.SkipUnlessProviderIs("gce", "gke")

		ReplicaSetServeImageOrFail(f, "private", "b.gcr.io/k8s_authenticated_test/serve_hostname:v1.4")
	})

	It("should surface a failure condition on a common issue like exceeded quota", func() {
		rsConditionCheck(f)
	})
})

// A basic test to check the deployment of an image using a ReplicaSet. The
// image serves its hostname which is checked for each replica.
func ReplicaSetServeImageOrFail(f *framework.Framework, test string, image string) {
	name := "my-hostname-" + test + "-" + string(uuid.NewUUID())
	replicas := int32(2)

	// Create a ReplicaSet for a service that serves its hostname.
	// The source for the Docker containter kubernetes/serve_hostname is
	// in contrib/for-demos/serve_hostname
	By(fmt.Sprintf("Creating ReplicaSet %s", name))
	rs, err := f.ClientSet.Extensions().ReplicaSets(f.Namespace.Name).Create(&extensions.ReplicaSet{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{
				"name": name,
			}},
			Template: v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  name,
							Image: image,
							Ports: []v1.ContainerPort{{ContainerPort: 9376}},
						},
					},
				},
			},
		},
	})
	Expect(err).NotTo(HaveOccurred())
	// Cleanup the ReplicaSet when we are done.
	defer func() {
		// Resize the ReplicaSet to zero to get rid of pods.
		if err := framework.DeleteReplicaSet(f.ClientSet, f.InternalClientset, f.Namespace.Name, rs.Name); err != nil {
			framework.Logf("Failed to cleanup ReplicaSet %v: %v.", rs.Name, err)
		}
	}()

	// List the pods, making sure we observe all the replicas.
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))

	pods, err := framework.PodsCreated(f.ClientSet, f.Namespace.Name, name, replicas)
	Expect(err).NotTo(HaveOccurred())

	By("Ensuring each pod is running")

	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		err = f.WaitForPodRunning(pod.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	// Verify that something is listening.
	By("Trying to dial each unique pod")
	retryTimeout := 2 * time.Minute
	retryInterval := 5 * time.Second
	err = wait.Poll(retryInterval, retryTimeout, framework.PodProxyResponseChecker(f.ClientSet, f.Namespace.Name, label, name, true, pods).CheckAllResponses)
	if err != nil {
		framework.Failf("Did not get expected responses within the timeout period of %.2f seconds.", retryTimeout.Seconds())
	}
}

// 1. Create a quota restricting pods in the current namespace to 2.
// 2. Create a replica set that wants to run 3 pods.
// 3. Check replica set conditions for a ReplicaFailure condition.
// 4. Scale down the replica set and observe the condition is gone.
func rsConditionCheck(f *framework.Framework) {
	c := f.ClientSet
	namespace := f.Namespace.Name
	name := "condition-test"

	By(fmt.Sprintf("Creating quota %q that allows only two pods to run in the current namespace", name))
	quota := newPodQuota(name, "2")
	_, err := c.Core().ResourceQuotas(namespace).Create(quota)
	Expect(err).NotTo(HaveOccurred())

	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		quota, err = c.Core().ResourceQuotas(namespace).Get(name)
		if err != nil {
			return false, err
		}
		quantity := resource.MustParse("2")
		podQuota := quota.Status.Hard[v1.ResourcePods]
		return (&podQuota).Cmp(quantity) == 0, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("resource quota %q never synced", name)
	}
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Creating replica set %q that asks for more than the allowed pod quota", name))
	rs := newRS(name, 3, map[string]string{"name": name}, nginxImageName, nginxImage)
	rs, err = c.Extensions().ReplicaSets(namespace).Create(rs)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Checking replica set %q has the desired failure condition set", name))
	generation := rs.Generation
	conditions := rs.Status.Conditions
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		rs, err = c.Extensions().ReplicaSets(namespace).Get(name)
		if err != nil {
			return false, err
		}

		if generation > rs.Status.ObservedGeneration {
			return false, nil
		}
		conditions = rs.Status.Conditions

		cond := replicaset.GetCondition(rs.Status, extensions.ReplicaSetReplicaFailure)
		return cond != nil, nil

	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("rs controller never added the failure condition for replica set %q: %#v", name, conditions)
	}
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Scaling down replica set %q to satisfy pod quota", name))
	rs, err = framework.UpdateReplicaSetWithRetries(c, namespace, name, func(update *extensions.ReplicaSet) {
		x := int32(2)
		update.Spec.Replicas = &x
	})
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Checking replica set %q has no failure condition set", name))
	generation = rs.Generation
	conditions = rs.Status.Conditions
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		rs, err = c.Extensions().ReplicaSets(namespace).Get(name)
		if err != nil {
			return false, err
		}

		if generation > rs.Status.ObservedGeneration {
			return false, nil
		}
		conditions = rs.Status.Conditions

		cond := replicaset.GetCondition(rs.Status, extensions.ReplicaSetReplicaFailure)
		return cond == nil, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("rs controller never removed the failure condition for rs %q: %#v", name, conditions)
	}
	Expect(err).NotTo(HaveOccurred())
}
