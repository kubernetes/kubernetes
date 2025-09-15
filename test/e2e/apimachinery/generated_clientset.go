/*
Copyright 2014 The Kubernetes Authors.

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
	"context"
	"strconv"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

func testingPod(name, value string) v1.Pod {
	return v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": "foo",
				"time": value,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "nginx",
					Image: imageutils.GetE2EImage(imageutils.Nginx),
					Ports: []v1.ContainerPort{{ContainerPort: 80}},
					LivenessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path: "/index.html",
								Port: intstr.FromInt32(8080),
							},
						},
						InitialDelaySeconds: 30,
					},
				},
			},
		},
	}
}

func observeCreation(w watch.Interface) {
	select {
	case event := <-w.ResultChan():
		if event.Type != watch.Added {
			framework.Failf("Failed to observe the creation: %v", event)
		}
	case <-time.After(30 * time.Second):
		framework.Failf("Timeout while waiting for observing the creation")
	}
}

func observerUpdate(w watch.Interface, expectedUpdate func(runtime.Object) bool) {
	timer := time.After(30 * time.Second)
	updated := false
	timeout := false
	for !updated && !timeout {
		select {
		case event := <-w.ResultChan():
			if event.Type == watch.Modified {
				if expectedUpdate(event.Object) {
					updated = true
				}
			}
		case <-timer:
			timeout = true
		}
	}
	if !updated {
		framework.Failf("Failed to observe pod update")
	}
}

var _ = SIGDescribe("Generated clientset", func() {
	f := framework.NewDefaultFramework("clientset")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.It("should create pods, set the deletionTimestamp and deletionGracePeriodSeconds of the pod", func(ctx context.Context) {
		podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
		ginkgo.By("constructing the pod")
		name := "pod" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		podCopy := testingPod(name, value)
		pod := &podCopy
		ginkgo.By("setting up watch")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value})).String()
		options := metav1.ListOptions{LabelSelector: selector}
		pods, err := podClient.List(ctx, options)
		if err != nil {
			framework.Failf("Failed to query for pods: %v", err)
		}
		gomega.Expect(pods.Items).To(gomega.BeEmpty())
		options = metav1.ListOptions{
			LabelSelector:   selector,
			ResourceVersion: pods.ListMeta.ResourceVersion,
		}
		w, err := podClient.Watch(ctx, options)
		if err != nil {
			framework.Failf("Failed to set up watch: %v", err)
		}

		ginkgo.By("creating the pod")
		pod, err = podClient.Create(ctx, pod, metav1.CreateOptions{})
		if err != nil {
			framework.Failf("Failed to create pod: %v", err)
		}

		ginkgo.By("verifying the pod is in kubernetes")
		options = metav1.ListOptions{
			LabelSelector:   selector,
			ResourceVersion: pod.ResourceVersion,
		}
		pods, err = podClient.List(ctx, options)
		if err != nil {
			framework.Failf("Failed to query for pods: %v", err)
		}
		gomega.Expect(pods.Items).To(gomega.HaveLen(1))

		ginkgo.By("verifying pod creation was observed")
		observeCreation(w)

		// We need to wait for the pod to be scheduled, otherwise the deletion
		// will be carried out immediately rather than gracefully.
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))

		ginkgo.By("deleting the pod gracefully")
		gracePeriod := int64(31)
		if err := podClient.Delete(ctx, pod.Name, *metav1.NewDeleteOptions(gracePeriod)); err != nil {
			framework.Failf("Failed to delete pod: %v", err)
		}

		ginkgo.By("verifying the deletionTimestamp and deletionGracePeriodSeconds of the pod is set")
		observerUpdate(w, func(obj runtime.Object) bool {
			pod := obj.(*v1.Pod)
			return pod.ObjectMeta.DeletionTimestamp != nil && *pod.ObjectMeta.DeletionGracePeriodSeconds == gracePeriod
		})
	})
})

func newTestingCronJob(name string, value string) *batchv1.CronJob {
	parallelism := int32(1)
	completions := int32(1)
	return &batchv1.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"time": value,
			},
		},
		Spec: batchv1.CronJobSpec{
			Schedule:          "*/1 * * * ?",
			ConcurrencyPolicy: batchv1.AllowConcurrent,
			JobTemplate: batchv1.JobTemplateSpec{
				Spec: batchv1.JobSpec{
					Parallelism: &parallelism,
					Completions: &completions,
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyOnFailure,
							Volumes: []v1.Volume{
								{
									Name: "data",
									VolumeSource: v1.VolumeSource{
										EmptyDir: &v1.EmptyDirVolumeSource{},
									},
								},
							},
							Containers: []v1.Container{
								{
									Name:  "c",
									Image: imageutils.GetE2EImage(imageutils.BusyBox),
									VolumeMounts: []v1.VolumeMount{
										{
											MountPath: "/data",
											Name:      "data",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}
}

var _ = SIGDescribe("Generated clientset", func() {
	f := framework.NewDefaultFramework("clientset")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should create v1 cronJobs, delete cronJobs, watch cronJobs", func(ctx context.Context) {
		cronJobClient := f.ClientSet.BatchV1().CronJobs(f.Namespace.Name)
		ginkgo.By("constructing the cronJob")
		name := "cronjob" + string(uuid.NewUUID())
		value := strconv.Itoa(time.Now().Nanosecond())
		cronJob := newTestingCronJob(name, value)
		ginkgo.By("setting up watch")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value})).String()
		options := metav1.ListOptions{LabelSelector: selector}
		cronJobs, err := cronJobClient.List(ctx, options)
		if err != nil {
			framework.Failf("Failed to query for cronJobs: %v", err)
		}
		gomega.Expect(cronJobs.Items).To(gomega.BeEmpty())
		options = metav1.ListOptions{
			LabelSelector:   selector,
			ResourceVersion: cronJobs.ListMeta.ResourceVersion,
		}
		w, err := cronJobClient.Watch(ctx, options)
		if err != nil {
			framework.Failf("Failed to set up watch: %v", err)
		}

		ginkgo.By("creating the cronJob")
		cronJob, err = cronJobClient.Create(ctx, cronJob, metav1.CreateOptions{})
		if err != nil {
			framework.Failf("Failed to create cronJob: %v", err)
		}

		ginkgo.By("verifying the cronJob is in kubernetes")
		options = metav1.ListOptions{
			LabelSelector:   selector,
			ResourceVersion: cronJob.ResourceVersion,
		}
		cronJobs, err = cronJobClient.List(ctx, options)
		if err != nil {
			framework.Failf("Failed to query for cronJobs: %v", err)
		}
		gomega.Expect(cronJobs.Items).To(gomega.HaveLen(1))

		ginkgo.By("verifying cronJob creation was observed")
		observeCreation(w)

		ginkgo.By("deleting the cronJob")
		// Use DeletePropagationBackground so the CronJob is really gone when the call returns.
		propagationPolicy := metav1.DeletePropagationBackground
		if err := cronJobClient.Delete(ctx, cronJob.Name, metav1.DeleteOptions{PropagationPolicy: &propagationPolicy}); err != nil {
			framework.Failf("Failed to delete cronJob: %v", err)
		}

		options = metav1.ListOptions{LabelSelector: selector}
		cronJobs, err = cronJobClient.List(ctx, options)
		if err != nil {
			framework.Failf("Failed to list cronJobs to verify deletion: %v", err)
		}
		gomega.Expect(cronJobs.Items).To(gomega.BeEmpty())
	})
})
