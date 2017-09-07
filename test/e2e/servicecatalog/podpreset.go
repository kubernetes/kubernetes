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

package servicecatalog

import (
	"reflect"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	settings "k8s.io/api/settings/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = SIGDescribe("[Feature:PodPreset] PodPreset", func() {
	f := framework.NewDefaultFramework("podpreset")

	var podClient *framework.PodClient
	BeforeEach(func() {
		// only run on gce for the time being til we find an easier way to update
		// the admission controllers used on the others
		framework.SkipUnlessProviderIs("gce")
		podClient = f.PodClient()
	})

	// Simplest case: all pods succeed promptly
	It("should create a pod preset", func() {
		By("Creating a pod preset")

		pip := &settings.PodPreset{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "hello",
				Namespace: f.Namespace.Name,
			},
			Spec: settings.PodPresetSpec{
				Selector: metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "security",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"S2"},
						},
					},
				},
				Volumes: []v1.Volume{{Name: "vol", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}},
				VolumeMounts: []v1.VolumeMount{
					{Name: "vol", MountPath: "/foo"},
				},
				Env: []v1.EnvVar{{Name: "abc", Value: "value"}, {Name: "ABC", Value: "value"}},
			},
		}

		_, err := createPodPreset(f.ClientSet, f.Namespace.Name, pip)
		Expect(err).NotTo(HaveOccurred())

		By("creating the pod")
		name := "pod-preset-pod"
		value := strconv.Itoa(time.Now().Nanosecond())
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: f.Namespace.Name,
				Labels: map[string]string{
					"name":     "foo",
					"time":     value,
					"security": "S2",
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "nginx",
						Image: imageutils.GetE2EImage(imageutils.NginxSlim),
					},
				},
			},
		}

		By("setting up watch")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pod")
		Expect(len(pods.Items)).To(Equal(0))
		options = metav1.ListOptions{
			LabelSelector:   selector.String(),
			ResourceVersion: pods.ListMeta.ResourceVersion,
		}
		w, err := podClient.Watch(options)
		Expect(err).NotTo(HaveOccurred(), "failed to set up watch")

		By("submitting the pod to kubernetes")
		podClient.Create(pod)

		By("verifying the pod is in kubernetes")
		selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options = metav1.ListOptions{LabelSelector: selector.String()}
		pods, err = podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pod")
		Expect(len(pods.Items)).To(Equal(1))

		By("verifying pod creation was observed")
		select {
		case event, _ := <-w.ResultChan():
			if event.Type != watch.Added {
				framework.Failf("Failed to observe pod creation: %v", event)
			}
		case <-time.After(framework.PodStartTimeout):
			framework.Failf("Timeout while waiting for pod creation")
		}

		// We need to wait for the pod to be running, otherwise the deletion
		// may be carried out immediately rather than gracefully.
		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

		By("ensuring pod is modified")
		// save the running pod
		pod, err = podClient.Get(pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to GET scheduled pod")

		// check the annotation is there
		if _, ok := pod.Annotations["podpreset.admission.kubernetes.io/podpreset-hello"]; !ok {
			framework.Failf("Annotation not found in pod annotations: \n%v\n", pod.Annotations)
		}

		// verify the env is the same
		if !reflect.DeepEqual(pip.Spec.Env, pod.Spec.Containers[0].Env) {
			framework.Failf("env of pod container does not match the env of the pip: expected %#v, got: %#v", pip.Spec.Env, pod.Spec.Containers[0].Env)
		}
	})

	It("should not modify the pod on conflict", func() {
		By("Creating a pod preset")

		pip := &settings.PodPreset{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "hello",
				Namespace: f.Namespace.Name,
			},
			Spec: settings.PodPresetSpec{
				Selector: metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "security",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"S2"},
						},
					},
				},
				Volumes: []v1.Volume{{Name: "vol", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}}},
				VolumeMounts: []v1.VolumeMount{
					{Name: "vol", MountPath: "/foo"},
				},
				Env: []v1.EnvVar{{Name: "abc", Value: "value"}, {Name: "ABC", Value: "value"}},
			},
		}

		_, err := createPodPreset(f.ClientSet, f.Namespace.Name, pip)
		Expect(err).NotTo(HaveOccurred())

		By("creating the pod")
		name := "pod-preset-pod"
		value := strconv.Itoa(time.Now().Nanosecond())
		originalPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: f.Namespace.Name,
				Labels: map[string]string{
					"name":     "foo",
					"time":     value,
					"security": "S2",
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "nginx",
						Image: imageutils.GetE2EImage(imageutils.NginxSlim),
						Env:   []v1.EnvVar{{Name: "abc", Value: "value2"}, {Name: "ABC", Value: "value"}},
					},
				},
			},
		}

		By("setting up watch")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		pods, err := podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pod")
		Expect(len(pods.Items)).To(Equal(0))
		options = metav1.ListOptions{
			LabelSelector:   selector.String(),
			ResourceVersion: pods.ListMeta.ResourceVersion,
		}
		w, err := podClient.Watch(options)
		Expect(err).NotTo(HaveOccurred(), "failed to set up watch")

		By("submitting the pod to kubernetes")
		podClient.Create(originalPod)

		By("verifying the pod is in kubernetes")
		selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
		options = metav1.ListOptions{LabelSelector: selector.String()}
		pods, err = podClient.List(options)
		Expect(err).NotTo(HaveOccurred(), "failed to query for pod")
		Expect(len(pods.Items)).To(Equal(1))

		By("verifying pod creation was observed")
		select {
		case event, _ := <-w.ResultChan():
			if event.Type != watch.Added {
				framework.Failf("Failed to observe pod creation: %v", event)
			}
		case <-time.After(framework.PodStartTimeout):
			framework.Failf("Timeout while waiting for pod creation")
		}

		// We need to wait for the pod to be running, otherwise the deletion
		// may be carried out immediately rather than gracefully.
		framework.ExpectNoError(f.WaitForPodRunning(originalPod.Name))

		By("ensuring pod is modified")
		// save the running pod
		pod, err := podClient.Get(originalPod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "failed to GET scheduled pod")

		// check the annotation is not there
		if _, ok := pod.Annotations["podpreset.admission.kubernetes.io/podpreset-hello"]; ok {
			framework.Failf("Annotation found in pod annotations and should not be: \n%v\n", pod.Annotations)
		}

		// verify the env is the same
		if !reflect.DeepEqual(originalPod.Spec.Containers[0].Env, pod.Spec.Containers[0].Env) {
			framework.Failf("env of pod container does not match the env of the original pod: expected %#v, got: %#v", originalPod.Spec.Containers[0].Env, pod.Spec.Containers[0].Env)
		}
	})
})

func getPodPreset(c clientset.Interface, ns, name string) (*settings.PodPreset, error) {
	return c.Settings().PodPresets(ns).Get(name, metav1.GetOptions{})
}

func createPodPreset(c clientset.Interface, ns string, job *settings.PodPreset) (*settings.PodPreset, error) {
	return c.Settings().PodPresets(ns).Create(job)
}

func updatePodPreset(c clientset.Interface, ns string, job *settings.PodPreset) (*settings.PodPreset, error) {
	return c.Settings().PodPresets(ns).Update(job)
}

func deletePodPreset(c clientset.Interface, ns, name string) error {
	return c.Settings().PodPresets(ns).Delete(name, nil)
}
