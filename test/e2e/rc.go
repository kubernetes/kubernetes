/*
Copyright 2015 The Kubernetes Authors.

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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("ReplicationController", func() {
	f := framework.NewDefaultFramework("replication-controller")

	It("should serve a basic image on each replica with a public image [Conformance]", func() {
		ServeImageOrFail(f, "basic", "gcr.io/google_containers/serve_hostname:v1.4")
	})

	It("should serve a basic image on each replica with a private image", func() {
		// requires private images
		framework.SkipUnlessProviderIs("gce", "gke")

		ServeImageOrFail(f, "private", "b.gcr.io/k8s_authenticated_test/serve_hostname:v1.4")
	})

	It("should surface a failure condition on a common issue like exceeded quota", func() {
		rcConditionCheck(f)
	})
})

func newRC(rsName string, replicas int32, rcPodLabels map[string]string, imageName string, image string) *v1.ReplicationController {
	zero := int64(0)
	return &v1.ReplicationController{
		ObjectMeta: v1.ObjectMeta{
			Name: rsName,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Template: &v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: rcPodLabels,
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

// A basic test to check the deployment of an image using
// a replication controller. The image serves its hostname
// which is checked for each replica.
func ServeImageOrFail(f *framework.Framework, test string, image string) {
	name := "my-hostname-" + test + "-" + string(uuid.NewUUID())
	replicas := int32(2)

	// Create a replication controller for a service
	// that serves its hostname.
	// The source for the Docker containter kubernetes/serve_hostname is
	// in contrib/for-demos/serve_hostname
	By(fmt.Sprintf("Creating replication controller %s", name))
	controller, err := f.ClientSet.Core().ReplicationControllers(f.Namespace.Name).Create(&v1.ReplicationController{
		ObjectMeta: v1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Selector: map[string]string{
				"name": name,
			},
			Template: &v1.PodTemplateSpec{
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
	// Cleanup the replication controller when we are done.
	defer func() {
		// Resize the replication controller to zero to get rid of pods.
		if err := framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, controller.Name); err != nil {
			framework.Logf("Failed to cleanup replication controller %v: %v.", controller.Name, err)
		}
	}()

	// List the pods, making sure we observe all the replicas.
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))

	pods, err := framework.PodsCreated(f.ClientSet, f.Namespace.Name, name, replicas)

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
// 2. Create a replication controller that wants to run 3 pods.
// 3. Check replication controller conditions for a ReplicaFailure condition.
// 4. Relax quota or scale down the controller and observe the condition is gone.
func rcConditionCheck(f *framework.Framework) {
	c := f.ClientSet
	namespace := f.Namespace.Name
	name := "condition-test"

	By(fmt.Sprintf("Creating quota %q that allows only two pods to run in the current namespace", name))
	quota := newPodQuota(name, "2")
	_, err := c.Core().ResourceQuotas(namespace).Create(quota)
	Expect(err).NotTo(HaveOccurred())

	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		quota, err = c.Core().ResourceQuotas(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		podQuota := quota.Status.Hard[v1.ResourcePods]
		quantity := resource.MustParse("2")
		return (&podQuota).Cmp(quantity) == 0, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("resource quota %q never synced", name)
	}
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Creating rc %q that asks for more than the allowed pod quota", name))
	rc := newRC(name, 3, map[string]string{"name": name}, nginxImageName, nginxImage)
	rc, err = c.Core().ReplicationControllers(namespace).Create(rc)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Checking rc %q has the desired failure condition set", name))
	generation := rc.Generation
	conditions := rc.Status.Conditions
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		rc, err = c.Core().ReplicationControllers(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if generation > rc.Status.ObservedGeneration {
			return false, nil
		}
		conditions = rc.Status.Conditions

		cond := replication.GetCondition(rc.Status, v1.ReplicationControllerReplicaFailure)
		return cond != nil, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("rc manager never added the failure condition for rc %q: %#v", name, conditions)
	}
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Scaling down rc %q to satisfy pod quota", name))
	rc, err = framework.UpdateReplicationControllerWithRetries(c, namespace, name, func(update *v1.ReplicationController) {
		x := int32(2)
		update.Spec.Replicas = &x
	})
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Checking rc %q has no failure condition set", name))
	generation = rc.Generation
	conditions = rc.Status.Conditions
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		rc, err = c.Core().ReplicationControllers(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if generation > rc.Status.ObservedGeneration {
			return false, nil
		}
		conditions = rc.Status.Conditions

		cond := replication.GetCondition(rc.Status, v1.ReplicationControllerReplicaFailure)
		return cond == nil, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("rc manager never removed the failure condition for rc %q: %#v", name, conditions)
	}
	Expect(err).NotTo(HaveOccurred())
}
