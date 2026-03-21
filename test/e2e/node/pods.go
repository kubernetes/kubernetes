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

package node

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Pods Extended", func() {
	f := framework.NewDefaultFramework("pods")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Describe("Delete Grace Period", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			podClient = e2epod.NewPodClient(f)
		})

		/*
			Release: v1.15
			Testname: Pods, delete grace period
			Description: Create a pod, make sure it is running. Using the http client send a 'delete' with gracePeriodSeconds=30. Pod SHOULD get terminated within gracePeriodSeconds and removed from API server within a window.
		*/
		ginkgo.It("should be submitted and removed", func(ctx context.Context) {
			ginkgo.By("creating the pod")
			name := "pod-submit-remove-" + string(uuid.NewUUID())
			value := strconv.Itoa(time.Now().Nanosecond())
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, name, nil, nil, nil)
			pod.ObjectMeta.Labels = map[string]string{
				"name": "foo",
				"time": value,
			}

			ginkgo.By("setting up selector")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			pods, err := podClient.List(ctx, options)
			framework.ExpectNoError(err, "failed to query for pod")
			gomega.Expect(pods.Items).To(gomega.BeEmpty())

			ginkgo.By("submitting the pod to kubernetes")
			podClient.Create(ctx, pod)

			ginkgo.By("verifying the pod is in kubernetes")
			selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options = metav1.ListOptions{LabelSelector: selector.String()}
			pods, err = podClient.List(ctx, options)
			framework.ExpectNoError(err, "failed to query for pod")
			gomega.Expect(pods.Items).To(gomega.HaveLen(1))

			// We need to wait for the pod to be running, otherwise the deletion
			// may be carried out immediately rather than gracefully.
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
			// save the running pod
			pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to GET scheduled pod")

			ginkgo.By("deleting the pod gracefully")
			var lastPod v1.Pod
			var statusCode int
			err = f.ClientSet.CoreV1().RESTClient().Delete().AbsPath("/api/v1/namespaces", pod.Namespace, "pods", pod.Name).Param("gracePeriodSeconds", "30").Do(ctx).StatusCode(&statusCode).Into(&lastPod)
			framework.ExpectNoError(err, "failed to use http client to send delete")
			gomega.Expect(statusCode).To(gomega.Equal(http.StatusOK), "failed to delete gracefully by client request")

			ginkgo.By("verifying the kubelet observed the termination notice")

			// allow up to 3x grace period (which allows process termination)
			// for the kubelet to remove from api.  need to follow-up on if this
			// latency between termination and reportal can be isolated further.
			start := time.Now()
			err = wait.Poll(time.Second*5, time.Second*30*3, func() (bool, error) {
				podList, err := e2ekubelet.GetKubeletPods(ctx, f.ClientSet, pod.Spec.NodeName)
				if err != nil {
					framework.Logf("Unable to retrieve kubelet pods for node %v: %v", pod.Spec.NodeName, err)
					return false, nil
				}
				for _, kubeletPod := range podList.Items {
					if pod.Name != kubeletPod.Name || pod.Namespace != kubeletPod.Namespace {
						continue
					}
					if kubeletPod.ObjectMeta.DeletionTimestamp == nil {
						framework.Logf("deletion has not yet been observed")
						return false, nil
					}
					data, _ := json.Marshal(kubeletPod)
					framework.Logf("start=%s, now=%s, kubelet pod: %s", start, time.Now(), string(data))
					return false, nil
				}
				framework.Logf("no pod exists with the name we were looking for, assuming the termination request was observed and completed")
				return true, nil
			})
			framework.ExpectNoError(err, "kubelet never observed the termination notice")

			gomega.Expect(lastPod.DeletionTimestamp).ToNot(gomega.BeNil())
			gomega.Expect(lastPod.Spec.TerminationGracePeriodSeconds).ToNot(gomega.BeZero())

			selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options = metav1.ListOptions{LabelSelector: selector.String()}
			pods, err = podClient.List(ctx, options)
			framework.ExpectNoError(err, "failed to query for pods")
			gomega.Expect(pods.Items).To(gomega.BeEmpty())
		})
	})

	ginkgo.Describe("Pods Set QOS Class", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			podClient = e2epod.NewPodClient(f)
		})

		/*
			Release: v1.9
			Testname: Pods, QOS
			Description:  Create a Pod with CPU and Memory request and limits. Pod status MUST have QOSClass set to PodQOSGuaranteed.
		*/
		framework.ConformanceIt("should be set on Pods with matching resource requests and limits for memory and cpu", func(ctx context.Context) {
			ginkgo.By("creating the pod")
			name := "pod-qos-class-" + string(uuid.NewUUID())
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
					Labels: map[string]string{
						"name": name,
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "agnhost",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Args:  []string{"pause"},
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
								Requests: v1.ResourceList{
									v1.ResourceCPU:    resource.MustParse("100m"),
									v1.ResourceMemory: resource.MustParse("100Mi"),
								},
							},
						},
					},
				},
			}

			ginkgo.By("submitting the pod to kubernetes")
			podClient.Create(ctx, pod)

			ginkgo.By("verifying QOS class is set on the pod")
			pod, err := podClient.Get(ctx, name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to query for pod")
			gomega.Expect(pod.Status.QOSClass).To(gomega.Equal(v1.PodQOSGuaranteed))
		})
	})

	ginkgo.Describe("Pod Container lifecycle", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			podClient = e2epod.NewPodClient(f)
		})

		ginkgo.It("should not create extra sandbox if all containers are done", func(ctx context.Context) {
			ginkgo.By("creating the pod that should always exit 0")

			name := "pod-always-succeed" + string(uuid.NewUUID())
			image := imageutils.GetE2EImage(imageutils.BusyBox)
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
					InitContainers: []v1.Container{
						{
							Name:  "foo",
							Image: image,
							Command: []string{
								"/bin/true",
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:  "bar",
							Image: image,
							Command: []string{
								"/bin/true",
							},
						},
					},
				},
			}

			ginkgo.By("submitting the pod to kubernetes")
			createdPod := podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})

			framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))

			var eventList *v1.EventList
			var err error
			ginkgo.By("Getting events about the pod")
			framework.ExpectNoError(wait.Poll(time.Second*2, time.Second*60, func() (bool, error) {
				selector := fields.Set{
					"involvedObject.kind":      "Pod",
					"involvedObject.uid":       string(createdPod.UID),
					"involvedObject.namespace": f.Namespace.Name,
					"source":                   "kubelet",
				}.AsSelector().String()
				options := metav1.ListOptions{FieldSelector: selector}
				eventList, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, options)
				if err != nil {
					return false, err
				}
				if len(eventList.Items) > 0 {
					return true, nil
				}
				return false, nil
			}))

			ginkgo.By("Checking events about the pod")
			for _, event := range eventList.Items {
				if event.Reason == events.SandboxChanged {
					framework.Fail("Unexpected SandboxChanged event")
				}
			}
		})

		ginkgo.It("evicted pods should be terminal", func(ctx context.Context) {
			ginkgo.By("creating the pod that should be evicted")

			name := "pod-should-be-evicted" + string(uuid.NewUUID())
			image := imageutils.GetE2EImage(imageutils.BusyBox)
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
					Containers: []v1.Container{
						{
							Name:  "bar",
							Image: image,
							Command: []string{
								"/bin/sh", "-c", "sleep 10; dd if=/dev/zero of=file bs=1M count=10; sleep 10000",
							},
							Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{
									"ephemeral-storage": resource.MustParse("5Mi"),
								},
							}},
					},
				},
			}

			ginkgo.By("submitting the pod to kubernetes")
			podClient.Create(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})

			// Intentionally increase the timeout to ensure the metrics availability required for this test.
			err := e2epod.WaitForPodTerminatedInNamespaceTimeout(ctx, f.ClientSet, pod.Name, "Evicted", f.Namespace.Name, 10*time.Minute)
			if err != nil {
				framework.Failf("error waiting for pod to be evicted: %v", err)
			}

		})
	})

	ginkgo.Describe("Pod TerminationGracePeriodSeconds is negative", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			podClient = e2epod.NewPodClient(f)
		})

		ginkgo.It("pod with negative grace period", func(ctx context.Context) {
			name := "pod-negative-grace-period" + string(uuid.NewUUID())
			image := imageutils.GetE2EImage(imageutils.BusyBox)
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
					Containers: []v1.Container{
						{
							Name:  "foo",
							Image: image,
							Command: []string{
								"/bin/sh", "-c", "sleep 10000",
							},
						},
					},
					TerminationGracePeriodSeconds: ptr.To[int64](-1),
				},
			}

			ginkgo.By("submitting the pod to kubernetes")
			podClient.Create(ctx, pod)

			pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to query for pod")

			if pod.Spec.TerminationGracePeriodSeconds == nil {
				framework.Failf("pod spec TerminationGracePeriodSeconds is nil")
			}

			if *pod.Spec.TerminationGracePeriodSeconds != 1 {
				framework.Failf("pod spec TerminationGracePeriodSeconds is not 1: %d", *pod.Spec.TerminationGracePeriodSeconds)
			}

			// retry if the TerminationGracePeriodSeconds is overrided
			// see more in https://github.com/kubernetes/kubernetes/pull/115606
			err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
				pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "failed to query for pod")
				ginkgo.By("updating the pod to have a negative TerminationGracePeriodSeconds")
				pod.Spec.TerminationGracePeriodSeconds = ptr.To[int64](-1)
				_, err = podClient.PodInterface.Update(ctx, pod, metav1.UpdateOptions{})
				return err
			})
			framework.ExpectNoError(err, "failed to update pod")

			pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to query for pod")

			if pod.Spec.TerminationGracePeriodSeconds == nil {
				framework.Failf("pod spec TerminationGracePeriodSeconds is nil")
			}

			if *pod.Spec.TerminationGracePeriodSeconds != 1 {
				framework.Failf("pod spec TerminationGracePeriodSeconds is not 1: %d", *pod.Spec.TerminationGracePeriodSeconds)
			}

			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})
		})
	})
})

var _ = SIGDescribe("Pods Extended (pod generation)", func() {
	f := framework.NewDefaultFramework("pods")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Describe("Pod Generation", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			podClient = e2epod.NewPodClient(f)
		})

		/*
			Release: v1.35
			Testname: Pods Generation, updates
			Description: Create a Pod, and perform a few updates, ensuring that the pod's metadata.generation and status.observedGeneration are updated as expected.
		*/
		framework.ConformanceIt("pod generation should start at 1 and increment per update [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
			ginkgo.By("creating the pod")
			podName := "pod-generation-" + string(uuid.NewUUID())
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, podName, nil, nil, nil)
			pod.Spec.InitContainers = []v1.Container{{
				Name:  "init-container",
				Image: imageutils.GetE2EImage(imageutils.BusyBox),
			}}

			ginkgo.By("submitting the pod to kubernetes")
			pod = podClient.CreateSync(ctx, pod)
			gomega.Expect(pod.Generation).To(gomega.BeEquivalentTo(1))
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})

			ginkgo.By("verifying pod generation bumps as expected")
			tests := []struct {
				name                 string
				updateFn             func(*v1.Pod)
				expectGenerationBump bool
			}{
				{
					name:                 "empty update",
					updateFn:             func(pod *v1.Pod) {},
					expectGenerationBump: false,
				},

				{
					name: "updating Tolerations to trigger generation bump",
					updateFn: func(pod *v1.Pod) {
						pod.Spec.Tolerations = []v1.Toleration{
							{
								Key:      "foo-" + string(uuid.NewUUID()),
								Operator: v1.TolerationOpEqual,
								Value:    "bar",
								Effect:   v1.TaintEffectNoSchedule,
							},
						}
					},
					expectGenerationBump: true,
				},

				{
					name: "updating ActiveDeadlineSeconds to trigger generation bump",
					updateFn: func(pod *v1.Pod) {
						int5000 := int64(5000)
						pod.Spec.ActiveDeadlineSeconds = &int5000
					},
					expectGenerationBump: true,
				},

				{
					name: "updating container image to trigger generation bump",
					updateFn: func(pod *v1.Pod) {
						pod.Spec.Containers[0].Image = imageutils.GetE2EImage(imageutils.Nginx)
					},
					expectGenerationBump: true,
				},

				{
					name: "updating initContainer image to trigger generation bump",
					updateFn: func(pod *v1.Pod) {
						pod.Spec.InitContainers[0].Image = imageutils.GetE2EImage(imageutils.Pause)
					},
					expectGenerationBump: true,
				},

				{
					name: "updates to pod metadata should not trigger generation bump",
					updateFn: func(pod *v1.Pod) {
						pod.SetAnnotations(map[string]string{"key": "value"})
					},
					expectGenerationBump: false,
				},

				{
					name: "pod generation updated by client should be ignored",
					updateFn: func(pod *v1.Pod) {
						pod.SetGeneration(1)
					},
					expectGenerationBump: false,
				},
			}

			expectedPodGeneration := int64(1)
			for _, test := range tests {
				ginkgo.By(test.name)
				podClient.Update(ctx, podName, test.updateFn)
				pod, err := podClient.Get(ctx, podName, metav1.GetOptions{})
				framework.ExpectNoError(err, "failed to query for pod")
				if test.expectGenerationBump {
					expectedPodGeneration++
				}
				gomega.Expect(pod.Generation).To(gomega.BeEquivalentTo(expectedPodGeneration))
				framework.ExpectNoError(e2epod.WaitForPodObservedGeneration(ctx, f.ClientSet, f.Namespace.Name, pod.Name, expectedPodGeneration, 20*time.Second))
			}
		})

		/*
			Release: v1.35
			Testname: Pods Generation, graceful delete
			Description: Create a Pod, ensure that triggering a graceful delete causes the generation to be updated.
		*/
		framework.ConformanceIt("custom-set generation on new pods and graceful delete", func(ctx context.Context) {
			ginkgo.By("creating the pod")
			name := "pod-generation-" + string(uuid.NewUUID())
			value := strconv.Itoa(time.Now().Nanosecond())
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, name, nil, nil, nil)
			pod.ObjectMeta.Labels = map[string]string{
				"time": value,
			}
			pod.SetGeneration(100)

			ginkgo.By("submitting the pod to kubernetes")
			pod = podClient.CreateSync(ctx, pod)

			ginkgo.By("verifying the new pod's generation is 1")
			gomega.Expect(pod.Generation).To(gomega.BeEquivalentTo(1))

			ginkgo.By("issue a graceful delete to trigger generation bump")
			// We need to wait for the pod to be running, otherwise the deletion
			// may be carried out immediately rather than gracefully.
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name))
			pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to GET scheduled pod")

			var lastPod v1.Pod
			var statusCode int
			// Set gracePeriodSeconds to 60 to give us time to verify the generation bump.
			err = f.ClientSet.CoreV1().RESTClient().Delete().AbsPath("/api/v1/namespaces", pod.Namespace, "pods", pod.Name).Param("gracePeriodSeconds", "60").Do(ctx).StatusCode(&statusCode).Into(&lastPod)
			framework.ExpectNoError(err, "failed to use http client to send delete")
			gomega.Expect(statusCode).To(gomega.Equal(http.StatusOK), "failed to delete gracefully by client request")

			ginkgo.By("verifying the pod generation was bumped")
			pod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to query for pod")
			gomega.Expect(pod.Generation).To(gomega.BeEquivalentTo(2))
		})

		/*
			Release: v1.35
			Testname: Pods Generation, 500 updates
			Description: Create a Pod, issue 499 podSpec updates and verify generation and observedGeneration eventually converge to 500.
		*/
		framework.ConformanceIt("issue 500 podspec updates and verify generation and observedGeneration eventually converge [MinimumKubeletVersion:1.34]", func(ctx context.Context) {
			ginkgo.By("creating the pod")
			name := "pod-generation-" + string(uuid.NewUUID())
			value := strconv.Itoa(time.Now().Nanosecond())
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, name, nil, nil, nil)
			pod.ObjectMeta.Labels = map[string]string{
				"time": value,
			}
			pod.Spec.ActiveDeadlineSeconds = ptr.To[int64](5000)

			ginkgo.By("submitting the pod to kubernetes")
			pod = podClient.CreateSync(ctx, pod)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})

			for range 499 {
				podClient.Update(ctx, pod.Name, func(pod *v1.Pod) {
					*pod.Spec.ActiveDeadlineSeconds--
				})
			}

			// Verify pod observedGeneration converges to the expected generation.
			expectedPodGeneration := int64(500)
			framework.ExpectNoError(e2epod.WaitForPodObservedGeneration(ctx, f.ClientSet, f.Namespace.Name, pod.Name, expectedPodGeneration, framework.PodStartTimeout))

			// Verify pod generation converges to the expected generation.
			pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to query for pod")
			gomega.Expect(pod.Generation).To(gomega.BeEquivalentTo(expectedPodGeneration))
		})
		ginkgo.It("pod observedGeneration field set in pod conditions", func(ctx context.Context) {
			ginkgo.By("creating the pod")
			name := "pod-generation-" + string(uuid.NewUUID())
			pod := e2epod.NewAgnhostPod(f.Namespace.Name, name, nil, nil, nil)

			// Set the pod image to something that doesn't exist to induce a pull error
			// to start with.
			agnImage := pod.Spec.Containers[0].Image
			pod.Spec.Containers[0].Image = "localhost/some-image-that-doesnt-exist"

			ginkgo.By("submitting the pod to kubernetes")
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			ginkgo.DeferCleanup(func(ctx context.Context) error {
				ginkgo.By("deleting the pod")
				return podClient.Delete(ctx, pod.Name, metav1.DeleteOptions{})
			})

			expectedPodConditions := []v1.PodConditionType{
				// add back once PodReadyToStartContainers feature GAs
				// v1.PodReadyToStartContainers
				v1.PodInitialized,
				v1.PodReady,
				v1.ContainersReady,
				v1.PodScheduled,
			}

			ginkgo.By("verifying the pod conditions have observedGeneration values")
			expectedObservedGeneration := int64(1)
			for _, condition := range expectedPodConditions {
				framework.ExpectNoError(e2epod.WaitForPodConditionObservedGeneration(ctx, f.ClientSet, f.Namespace.Name, pod.Name, condition, expectedObservedGeneration, framework.PodStartTimeout))
			}

			ginkgo.By("waiting for the container to fail to pull the image")
			// We need to wait for the container to fail to pull the image to avoid a race condition
			// where the pod is still being initialized by kubelet while the pod update is received.
			framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "image pull failure", framework.PodStartTimeout, func(pod *v1.Pod) (bool, error) {
				if len(pod.Status.ContainerStatuses) > 0 {
					status := pod.Status.ContainerStatuses[0]
					if status.State.Waiting != nil {
						reason := status.State.Waiting.Reason
						return reason == "ErrImagePull" || reason == "ImagePullBackOff", nil
					}
				}
				return false, nil
			}))

			ginkgo.By("updating pod to have a valid image")
			podClient.Update(ctx, pod.Name, func(pod *v1.Pod) {
				pod.Spec.Containers[0].Image = agnImage
			})
			expectedObservedGeneration++

			ginkgo.By("verifying the pod conditions have updated observedGeneration values")
			for _, condition := range expectedPodConditions {
				framework.ExpectNoError(e2epod.WaitForPodConditionObservedGeneration(ctx, f.ClientSet, f.Namespace.Name, pod.Name, condition, expectedObservedGeneration, framework.PodStartTimeout))
			}
		})
	})
})
