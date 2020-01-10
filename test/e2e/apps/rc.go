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

package apps

import (
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("ReplicationController", func() {
	f := framework.NewDefaultFramework("replication-controller")

	/*
		Release : v1.9
		Testname: Replication Controller, run basic image
		Description: Replication Controller MUST create a Pod with Basic Image and MUST run the service with the provided image. Image MUST be tested by dialing into the service listening through TCP, UDP and HTTP.
	*/

	framework.ConformanceIt("should serve a basic image on each replica with a public image ", func() {
		TestReplicationControllerServeImageOrFail(f, "basic", framework.ServeHostnameImage)
	})

	ginkgo.It("should serve a basic image on each replica with a private image", func() {
		// requires private images
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		privateimage := imageutils.GetConfig(imageutils.AgnhostPrivate)
		TestReplicationControllerServeImageOrFail(f, "private", privateimage.GetE2EImage())
	})

	/*
		Release : v1.15
		Testname: Replication Controller, check for issues like exceeding allocated quota
		Description: Attempt to create a Replication Controller with pods exceeding the namespace quota. The creation MUST fail
	*/
	framework.ConformanceIt("should surface a failure condition on a common issue like exceeded quota", func() {
		testReplicationControllerConditionCheck(f)
	})

	/*
		Release : v1.13
		Testname: Replication Controller, adopt matching pods
		Description: An ownerless Pod is created, then a Replication Controller (RC) is created whose label selector will match the Pod. The RC MUST either adopt the Pod or delete and replace it with a new Pod
	*/
	framework.ConformanceIt("should adopt matching pods on creation", func() {
		testRCAdoptMatchingOrphans(f)
	})

	/*
		Release : v1.13
		Testname: Replication Controller, release pods
		Description: A Replication Controller (RC) is created, and its Pods are created. When the labels on one of the Pods change to no longer match the RC's label selector, the RC MUST release the Pod and update the Pod's owner references.
	*/
	framework.ConformanceIt("should release no longer matching pods", func() {
		testRCReleaseControlledNotMatching(f)
	})
})

func newRC(rsName string, replicas int32, rcPodLabels map[string]string, imageName string, image string, args []string) *v1.ReplicationController {
	zero := int64(0)
	return &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name: rsName,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: rcPodLabels,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []v1.Container{
						{
							Name:  imageName,
							Image: image,
							Args:  args,
						},
					},
				},
			},
		},
	}
}

// TestReplicationControllerServeImageOrFail is a basic test to check
// the deployment of an image using a replication controller.
// The image serves its hostname which is checked for each replica.
func TestReplicationControllerServeImageOrFail(f *framework.Framework, test string, image string) {
	name := "my-hostname-" + test + "-" + string(uuid.NewUUID())
	replicas := int32(1)

	// Create a replication controller for a service
	// that serves its hostname.
	// The source for the Docker container kubernetes/serve_hostname is
	// in contrib/for-demos/serve_hostname
	ginkgo.By(fmt.Sprintf("Creating replication controller %s", name))
	newRC := newRC(name, replicas, map[string]string{"name": name}, name, image, []string{"serve-hostname"})
	newRC.Spec.Template.Spec.Containers[0].Ports = []v1.ContainerPort{{ContainerPort: 9376}}
	_, err := f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(newRC)
	framework.ExpectNoError(err)

	// Check that pods for the new RC were created.
	// TODO: Maybe switch PodsCreated to just check owner references.
	pods, err := e2epod.PodsCreated(f.ClientSet, f.Namespace.Name, name, replicas)
	framework.ExpectNoError(err)

	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	framework.Logf("Ensuring all pods for ReplicationController %q are running", name)
	running := int32(0)
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		err = f.WaitForPodRunning(pod.Name)
		if err != nil {
			updatePod, getErr := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
			if getErr == nil {
				err = fmt.Errorf("Pod %q never run (phase: %s, conditions: %+v): %v", updatePod.Name, updatePod.Status.Phase, updatePod.Status.Conditions, err)
			} else {
				err = fmt.Errorf("Pod %q never run: %v", pod.Name, err)
			}
		}
		framework.ExpectNoError(err)
		framework.Logf("Pod %q is running (conditions: %+v)", pod.Name, pod.Status.Conditions)
		running++
	}

	// Sanity check
	if running != replicas {
		framework.ExpectNoError(fmt.Errorf("unexpected number of running pods: %+v", pods.Items))
	}

	// Verify that something is listening.
	framework.Logf("Trying to dial the pod")
	retryTimeout := 2 * time.Minute
	retryInterval := 5 * time.Second
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	err = wait.Poll(retryInterval, retryTimeout, e2epod.NewProxyResponseChecker(f.ClientSet, f.Namespace.Name, label, name, true, pods).CheckAllResponses)
	if err != nil {
		framework.Failf("Did not get expected responses within the timeout period of %.2f seconds.", retryTimeout.Seconds())
	}
}

// 1. Create a quota restricting pods in the current namespace to 2.
// 2. Create a replication controller that wants to run 3 pods.
// 3. Check replication controller conditions for a ReplicaFailure condition.
// 4. Relax quota or scale down the controller and observe the condition is gone.
func testReplicationControllerConditionCheck(f *framework.Framework) {
	c := f.ClientSet
	namespace := f.Namespace.Name
	name := "condition-test"

	framework.Logf("Creating quota %q that allows only two pods to run in the current namespace", name)
	quota := newPodQuota(name, "2")
	_, err := c.CoreV1().ResourceQuotas(namespace).Create(quota)
	framework.ExpectNoError(err)

	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		quota, err = c.CoreV1().ResourceQuotas(namespace).Get(name, metav1.GetOptions{})
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
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Creating rc %q that asks for more than the allowed pod quota", name))
	rc := newRC(name, 3, map[string]string{"name": name}, WebserverImageName, WebserverImage, nil)
	rc, err = c.CoreV1().ReplicationControllers(namespace).Create(rc)
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Checking rc %q has the desired failure condition set", name))
	generation := rc.Generation
	conditions := rc.Status.Conditions
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		rc, err = c.CoreV1().ReplicationControllers(namespace).Get(name, metav1.GetOptions{})
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
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Scaling down rc %q to satisfy pod quota", name))
	rc, err = updateReplicationControllerWithRetries(c, namespace, name, func(update *v1.ReplicationController) {
		x := int32(2)
		update.Spec.Replicas = &x
	})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Checking rc %q has no failure condition set", name))
	generation = rc.Generation
	conditions = rc.Status.Conditions
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		rc, err = c.CoreV1().ReplicationControllers(namespace).Get(name, metav1.GetOptions{})
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
	framework.ExpectNoError(err)
}

func testRCAdoptMatchingOrphans(f *framework.Framework) {
	name := "pod-adoption"
	ginkgo.By(fmt.Sprintf("Given a Pod with a 'name' label %s is created", name))
	p := f.PodClient().CreateSync(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": name,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  name,
					Image: WebserverImage,
				},
			},
		},
	})

	ginkgo.By("When a replication controller with a matching selector is created")
	replicas := int32(1)
	rcSt := newRC(name, replicas, map[string]string{"name": name}, name, WebserverImage, nil)
	rcSt.Spec.Selector = map[string]string{"name": name}
	rc, err := f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(rcSt)
	framework.ExpectNoError(err)

	ginkgo.By("Then the orphan pod is adopted")
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		p2, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(p.Name, metav1.GetOptions{})
		// The Pod p should either be adopted or deleted by the RC
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		framework.ExpectNoError(err)
		for _, owner := range p2.OwnerReferences {
			if *owner.Controller && owner.UID == rc.UID {
				// pod adopted
				return true, nil
			}
		}
		// pod still not adopted
		return false, nil
	})
	framework.ExpectNoError(err)
}

func testRCReleaseControlledNotMatching(f *framework.Framework) {
	name := "pod-release"
	ginkgo.By("Given a ReplicationController is created")
	replicas := int32(1)
	rcSt := newRC(name, replicas, map[string]string{"name": name}, name, WebserverImage, nil)
	rcSt.Spec.Selector = map[string]string{"name": name}
	rc, err := f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(rcSt)
	framework.ExpectNoError(err)

	ginkgo.By("When the matched label of one of its pods change")
	pods, err := e2epod.PodsCreated(f.ClientSet, f.Namespace.Name, rc.Name, replicas)
	framework.ExpectNoError(err)

	p := pods.Items[0]
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		pod.Labels = map[string]string{"name": "not-matching-name"}
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Update(pod)
		if err != nil && apierrors.IsConflict(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
	framework.ExpectNoError(err)

	ginkgo.By("Then the pod is released")
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		p2, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		for _, owner := range p2.OwnerReferences {
			if *owner.Controller && owner.UID == rc.UID {
				// pod still belonging to the replication controller
				return false, nil
			}
		}
		// pod already released
		return true, nil
	})
	framework.ExpectNoError(err)
}

type updateRcFunc func(d *v1.ReplicationController)

// updateReplicationControllerWithRetries retries updating the given rc on conflict with the following steps:
// 1. Get latest resource
// 2. applyUpdate
// 3. Update the resource
func updateReplicationControllerWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateRcFunc) (*v1.ReplicationController, error) {
	var rc *v1.ReplicationController
	var updateErr error
	pollErr := wait.PollImmediate(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		var err error
		if rc, err = c.CoreV1().ReplicationControllers(namespace).Get(name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(rc)
		if rc, err = c.CoreV1().ReplicationControllers(namespace).Update(rc); err == nil {
			framework.Logf("Updating replication controller %q", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("couldn't apply the provided updated to rc %q: %v", name, updateErr)
	}
	return rc, pollErr
}
