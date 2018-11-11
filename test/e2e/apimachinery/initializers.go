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

package apimachinery

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	clientretry "k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = SIGDescribe("Initializers [Feature:Initializers]", func() {
	f := framework.NewDefaultFramework("initializers")

	// TODO: Add failure traps once we have JustAfterEach
	// See https://github.com/onsi/ginkgo/issues/303

	It("should be invisible to controllers by default", func() {
		ns := f.Namespace.Name
		c := f.ClientSet

		podName := "uninitialized-pod"
		framework.Logf("Creating pod %s", podName)

		ch := make(chan struct{})
		go func() {
			pod := newUninitializedPod(podName)
			_, err := c.CoreV1().Pods(ns).Create(pod)
			Expect(err).NotTo(HaveOccurred(), "failed to create pod %s in namespace: %s", podName, ns)
			close(ch)
		}()

		// wait to ensure the scheduler does not act on an uninitialized pod
		err := wait.PollImmediate(2*time.Second, 15*time.Second, func() (bool, error) {
			p, err := c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
			if err != nil {
				if errors.IsNotFound(err) {
					return false, nil
				}
				return false, err
			}
			return len(p.Spec.NodeName) > 0, nil
		})
		Expect(err).To(Equal(wait.ErrWaitTimeout))

		// verify that we can update an initializing pod
		pod, err := c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to get pod %s in namespace: %s", podName, ns)
		pod.Annotations = map[string]string{"update-1": "test"}
		pod, err = c.CoreV1().Pods(ns).Update(pod)
		Expect(err).NotTo(HaveOccurred(), "failed to update pod %s in namespace %s to: %+v", pod.Name, ns, pod)

		// verify the list call filters out uninitialized pods
		listOptions := metav1.ListOptions{IncludeUninitialized: true}
		pods, err := c.CoreV1().Pods(ns).List(listOptions)
		Expect(err).NotTo(HaveOccurred(), "failed to list pods in namespace: %s, given list options: %+v", ns, listOptions)
		Expect(pods.Items).To(HaveLen(1))
		pods, err = c.CoreV1().Pods(ns).List(metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to list pods in namespace: %s", ns)
		Expect(pods.Items).To(HaveLen(0))

		// clear initializers
		pod.Initializers = nil
		pod, err = c.CoreV1().Pods(ns).Update(pod)
		Expect(err).NotTo(HaveOccurred(), "failed to update pod %s in namespace %s to: %+v", pod.Name, ns, pod)

		// pod should now start running
		err = framework.WaitForPodRunningInNamespace(c, pod)
		Expect(err).NotTo(HaveOccurred(), "error while waiting for pod %s to go to Running phase in namespace: %s", pod.Name, pod.Namespace)

		// ensure create call returns
		<-ch

		// verify that we cannot start the pod initializing again
		pod, err = c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to get pod %s in namespace: %s", podName, ns)
		pod.Initializers = &metav1.Initializers{
			Pending: []metav1.Initializer{{Name: "Other"}},
		}
		_, err = c.CoreV1().Pods(ns).Update(pod)
		if !errors.IsInvalid(err) || !strings.Contains(err.Error(), "immutable") {
			Fail(fmt.Sprintf("expected invalid error: %v", err))
		}
	})

	It("should dynamically register and apply initializers to pods [Serial]", func() {
		ns := f.Namespace.Name
		c := f.ClientSet

		podName := "uninitialized-pod"
		framework.Logf("Creating pod %s", podName)

		// create and register an initializer
		initializerName := "pod.test.e2e.kubernetes.io"
		initializerConfigName := "e2e-test-initializer"
		initializerConfig := &v1alpha1.InitializerConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: initializerConfigName},
			Initializers: []v1alpha1.Initializer{
				{
					Name: initializerName,
					Rules: []v1alpha1.Rule{
						{APIGroups: []string{""}, APIVersions: []string{"*"}, Resources: []string{"pods"}},
					},
				},
			},
		}
		_, err := c.AdmissionregistrationV1alpha1().InitializerConfigurations().Create(initializerConfig)
		if errors.IsNotFound(err) {
			framework.Skipf("dynamic configuration of initializers requires the alpha admissionregistration.k8s.io group to be enabled")
		}
		Expect(err).NotTo(HaveOccurred(), "failed to create and register initializer with config: %+v", initializerConfig)

		// we must remove the initializer when the test is complete and ensure no pods are pending for that initializer
		defer cleanupInitializer(c, initializerConfigName, initializerName)

		// poller configuration is 1 second, wait at least that long
		time.Sleep(3 * time.Second)

		// run create that blocks
		ch := make(chan struct{})
		go func() {
			defer close(ch)
			pod := newInitPod(podName)
			_, err := c.CoreV1().Pods(ns).Create(pod)
			Expect(err).NotTo(HaveOccurred(), "failed to create pod %s in namespace: %s", podName, ns)
		}()

		// wait until the pod shows up uninitialized
		By("Waiting until the pod is visible to a client")
		var pod *v1.Pod
		err = wait.PollImmediate(2*time.Second, 15*time.Second, func() (bool, error) {
			pod, err = c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{IncludeUninitialized: true})
			if errors.IsNotFound(err) {
				return false, nil
			}
			if err != nil {
				return false, err
			}
			return true, nil
		})
		Expect(err).NotTo(HaveOccurred(), "failed to get pod %s from namespace: %s", podName, ns)
		Expect(pod.Initializers).NotTo(BeNil())
		Expect(pod.Initializers.Pending).To(HaveLen(1))
		Expect(pod.Initializers.Pending[0].Name).To(Equal(initializerName))

		// pretend we are an initializer
		By("Completing initialization")
		pod.Initializers = nil
		pod, err = c.CoreV1().Pods(ns).Update(pod)
		Expect(err).NotTo(HaveOccurred(), "failed to update pod %s in namespace %s to: %+v", pod.Name, ns, pod)

		// ensure create call returns
		<-ch

		// pod should now start running
		err = framework.WaitForPodRunningInNamespace(c, pod)
		Expect(err).NotTo(HaveOccurred(), "error while waiting for pod %s to go to Running phase in namespace: %s", pod.Name, pod.Namespace)

		// bypass initialization by explicitly passing an empty pending list
		By("Setting an empty initializer as an admin to bypass initialization")
		podName = "preinitialized-pod"
		pod = newUninitializedPod(podName)
		pod.Initializers.Pending = nil
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		Expect(err).NotTo(HaveOccurred(), "failed to create pod %s in namespace: %s", podName, ns)
		Expect(pod.Initializers).To(BeNil())

		// bypass initialization for mirror pods
		By("Creating a mirror pod that bypasses initialization")
		podName = "mirror-pod"
		pod = newInitPod(podName)
		pod.Annotations = map[string]string{
			v1.MirrorPodAnnotationKey: "true",
		}
		pod.Spec.NodeName = "node-does-not-yet-exist"
		pod, err = c.CoreV1().Pods(ns).Create(pod)
		Expect(err).NotTo(HaveOccurred(), "failed to create pod %s in namespace: %s", podName, ns)
		Expect(pod.Initializers).To(BeNil())
		Expect(pod.Annotations[v1.MirrorPodAnnotationKey]).To(Equal("true"))
	})

	It("don't cause replicaset controller creating extra pods if the initializer is not handled [Serial]", func() {
		ns := f.Namespace.Name
		c := f.ClientSet

		podName := "uninitialized-pod"
		framework.Logf("Creating pod %s", podName)

		// create and register an initializer, without setting up a controller to handle it.
		initializerName := "pod.test.e2e.kubernetes.io"
		initializerConfigName := "e2e-test-initializer"
		initializerConfig := &v1alpha1.InitializerConfiguration{
			ObjectMeta: metav1.ObjectMeta{Name: initializerConfigName},
			Initializers: []v1alpha1.Initializer{
				{
					Name: initializerName,
					Rules: []v1alpha1.Rule{
						{APIGroups: []string{""}, APIVersions: []string{"*"}, Resources: []string{"pods"}},
					},
				},
			},
		}
		_, err := c.AdmissionregistrationV1alpha1().InitializerConfigurations().Create(initializerConfig)
		if errors.IsNotFound(err) {
			framework.Skipf("dynamic configuration of initializers requires the alpha admissionregistration.k8s.io group to be enabled")
		}
		Expect(err).NotTo(HaveOccurred(), "failed to create and register initializer with config: %+v", initializerConfig)

		// we must remove the initializer when the test is complete and ensure no pods are pending for that initializer
		defer cleanupInitializer(c, initializerConfigName, initializerName)

		// poller configuration is 1 second, wait at least that long
		time.Sleep(3 * time.Second)

		// create a replicaset
		rs := newReplicaset()
		persistedRS, err := c.ExtensionsV1beta1().ReplicaSets(ns).Create(rs)
		Expect(err).NotTo(HaveOccurred(), "failed to create replicaset %s in namespace: %s", persistedRS.Name, ns)
		// wait for replicaset controller to confirm that it has handled the creation
		err = waitForRSObservedGeneration(c, persistedRS.Namespace, persistedRS.Name, persistedRS.Generation)
		Expect(err).NotTo(HaveOccurred(), "replicaset %s failed to observe generation: %d", persistedRS.Name, persistedRS.Generation)

		// update the replicaset spec to trigger a resync
		patch := []byte(`{"spec":{"minReadySeconds":5}}`)
		persistedRS, err = c.ExtensionsV1beta1().ReplicaSets(ns).Patch(persistedRS.Name, types.StrategicMergePatchType, patch)
		Expect(err).NotTo(HaveOccurred(), "failed to apply to replicaset %s in namespace %s a strategic merge patch: %s", persistedRS.Name, ns, patch)

		// wait for replicaset controller to confirm that it has handle the spec update
		err = waitForRSObservedGeneration(c, persistedRS.Namespace, persistedRS.Name, persistedRS.Generation)
		Expect(err).NotTo(HaveOccurred(), "replicaset %s failed to observe generation: %d", persistedRS.Name, persistedRS.Generation)

		// verify that the replicaset controller doesn't create extra pod
		selector, err := metav1.LabelSelectorAsSelector(persistedRS.Spec.Selector)
		Expect(err).NotTo(HaveOccurred(), "failed to convert label selector %+v of LabelSelector api type into a struct that implements labels.Selector", persistedRS.Spec.Selector)

		listOptions := metav1.ListOptions{
			LabelSelector:        selector.String(),
			IncludeUninitialized: true,
		}
		pods, err := c.CoreV1().Pods(ns).List(listOptions)
		Expect(err).NotTo(HaveOccurred(), "failed to list pods in namespace: %s, given list options: %+v", ns, listOptions)
		Expect(len(pods.Items)).Should(Equal(1))
	})

	It("will be set to nil if a patch removes the last pending initializer", func() {
		ns := f.Namespace.Name
		c := f.ClientSet

		podName := "to-be-patch-initialized-pod"
		framework.Logf("Creating pod %s", podName)

		// TODO: lower the timeout so that the server responds faster.
		_, err := c.CoreV1().Pods(ns).Create(newUninitializedPod(podName))
		if err != nil && !errors.IsTimeout(err) {
			framework.Failf("expect err to be timeout error, got %v", err)
		}
		uninitializedPod, err := c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to get pod %s in namespace: %s", podName, ns)
		Expect(uninitializedPod.Initializers).NotTo(BeNil())
		Expect(len(uninitializedPod.Initializers.Pending)).Should(Equal(1))

		patch := fmt.Sprintf(`{"metadata":{"initializers":{"pending":[{"$patch":"delete","name":"%s"}]}}}`, uninitializedPod.Initializers.Pending[0].Name)
		patchedPod, err := c.CoreV1().Pods(ns).Patch(uninitializedPod.Name, types.StrategicMergePatchType, []byte(patch))
		Expect(err).NotTo(HaveOccurred(), "failed to apply to pod %s in namespace %s a strategic merge patch: %s", uninitializedPod.Name, ns, patch)
		Expect(patchedPod.Initializers).To(BeNil())
	})
})

func newUninitializedPod(podName string) *v1.Pod {
	pod := newInitPod(podName)
	pod.Initializers = &metav1.Initializers{
		Pending: []metav1.Initializer{{Name: "test.k8s.io"}},
	}
	return pod
}

func newReplicaset() *v1beta1.ReplicaSet {
	name := "initializer-test-replicaset"
	replicas := int32(1)
	labels := map[string]string{"initializer-test": "single-replicaset"}
	return &v1beta1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1beta1.ReplicaSetSpec{
			Replicas: &replicas,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []v1.Container{
						{
							Name:  name + "-container",
							Image: imageutils.GetE2EImage(imageutils.Porter),
						},
					},
				},
			},
		},
	}
}

func newInitPod(podName string) *v1.Pod {
	containerName := fmt.Sprintf("%s-container", podName)
	port := 8080
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  containerName,
					Image: imageutils.GetE2EImage(imageutils.Porter),
					Env:   []v1.EnvVar{{Name: fmt.Sprintf("SERVE_PORT_%d", port), Value: "foo"}},
					Ports: []v1.ContainerPort{{ContainerPort: int32(port)}},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	return pod
}

// removeInitializersFromAllPods walks all pods and ensures they don't have the provided initializer,
// to guarantee completing the test doesn't block the entire cluster.
func removeInitializersFromAllPods(c clientset.Interface, initializerName string) {
	pods, err := c.CoreV1().Pods("").List(metav1.ListOptions{IncludeUninitialized: true})
	if err != nil {
		return
	}
	for _, p := range pods.Items {
		if p.Initializers == nil {
			continue
		}
		err := clientretry.RetryOnConflict(clientretry.DefaultRetry, func() error {
			pod, err := c.CoreV1().Pods(p.Namespace).Get(p.Name, metav1.GetOptions{IncludeUninitialized: true})
			if err != nil {
				if errors.IsNotFound(err) {
					return nil
				}
				return err
			}
			if pod.Initializers == nil {
				return nil
			}
			var updated []metav1.Initializer
			for _, pending := range pod.Initializers.Pending {
				if pending.Name != initializerName {
					updated = append(updated, pending)
				}
			}
			if len(updated) == len(pod.Initializers.Pending) {
				return nil
			}
			pod.Initializers.Pending = updated
			if len(updated) == 0 {
				pod.Initializers = nil
			}
			framework.Logf("Found initializer on pod %s in ns %s", pod.Name, pod.Namespace)
			_, err = c.CoreV1().Pods(p.Namespace).Update(pod)
			return err
		})
		if err != nil {
			framework.Logf("Unable to remove initializer from pod %s in ns %s: %v", p.Name, p.Namespace, err)
		}
	}
}

// remove the initializerConfig, and remove the initializer from all pods
func cleanupInitializer(c clientset.Interface, initializerConfigName, initializerName string) {
	if err := c.AdmissionregistrationV1alpha1().InitializerConfigurations().Delete(initializerConfigName, nil); err != nil && !errors.IsNotFound(err) {
		framework.Logf("got error on deleting %s", initializerConfigName)
	}
	// poller configuration is 1 second, wait at least that long
	time.Sleep(3 * time.Second)
	// clear our initializer from anyone who got it
	removeInitializersFromAllPods(c, initializerName)
}

// waits till the RS status.observedGeneration matches metadata.generation.
func waitForRSObservedGeneration(c clientset.Interface, ns, name string, generation int64) error {
	return wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		rs, err := c.ExtensionsV1beta1().ReplicaSets(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if generation > rs.Status.ObservedGeneration {
			return false, nil
		}
		return true, nil
	})
}
