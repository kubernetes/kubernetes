/*
Copyright 2023 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/utils/ptr"
)

const (
	LivenessPrefix  = "Liveness"
	PostStartPrefix = "PostStart"
	PreStopPrefix   = "PreStop"
	ReadinessPrefix = "Readiness"
	StartupPrefix   = "Startup"
)

var containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways

func prefixedName(namePrefix string, name string) string {
	return fmt.Sprintf("%s-%s", namePrefix, name)
}

var _ = SIGDescribe(framework.WithNodeConformance(), "Containers Lifecycle", func() {
	f := framework.NewDefaultFramework("containers-lifecycle-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should launch init container serially before a regular container", func() {

		init1 := "init-1"
		init2 := "init-2"
		init3 := "init-3"
		regular1 := "regular-1"

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "initcontainer-test-pod",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:  init1,
						Image: busyboxImage,
						Command: ExecCommand(init1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
					{
						Name:  init2,
						Image: busyboxImage,
						Command: ExecCommand(init2, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
					{
						Name:  init3,
						Image: busyboxImage,
						Command: ExecCommand(init3, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							StartDelay: 5,
							Delay:      1,
							ExitCode:   0,
						}),
						StartupProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{
										"test",
										"-f",
										"started",
									},
								},
							},
						},
					},
				},
			},
		}

		preparePod(podSpec)

		/// generates an out file output like:
		//
		// 1682076093 4905.79 init-1 Starting 0
		// 1682076093 4905.80 init-1 Started
		// 1682076093 4905.80 init-1 Delaying 1
		// 1682076094 4906.80 init-1 Exiting
		// 1682076095 4907.70 init-2 Starting 0
		// 1682076095 4907.71 init-2 Started
		// 1682076095 4907.71 init-2 Delaying 1
		// 1682076096 4908.71 init-2 Exiting
		// 1682076097 4909.74 init-3 Starting 0
		// 1682076097 4909.74 init-3 Started
		// 1682076097 4909.74 init-3 Delaying 1
		// 1682076098 4910.75 init-3 Exiting
		// 1682076099 4911.70 regular-1 Starting 5
		// 1682076104 4916.71 regular-1 Started
		// 1682076104 4916.71 regular-1 Delaying 1
		// 1682076105 4917.72 regular-1 Exiting

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to finish")
		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 1*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, podSpec)

		// which we then use to make assertions regarding container ordering
		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.StartsBefore(init1, init2))
		framework.ExpectNoError(results.ExitsBefore(init1, init2))

		framework.ExpectNoError(results.StartsBefore(init2, init3))
		framework.ExpectNoError(results.ExitsBefore(init2, init3))

		framework.ExpectNoError(results.StartsBefore(init3, regular1))
		framework.ExpectNoError(results.ExitsBefore(init3, regular1))
	})

	ginkgo.It("should not launch regular containers if an init container fails", func() {

		init1 := "init-1"
		regular1 := "regular-1"

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "initcontainer-test-pod-failure",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:  init1,
						Image: busyboxImage,
						Command: ExecCommand(init1, execCommand{
							Delay:    1,
							ExitCode: 1,
						}),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(podSpec)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to fail")
		err := e2epod.WaitForPodFailedReason(context.TODO(), f.ClientSet, podSpec, "", 1*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, podSpec)

		ginkgo.By("Analyzing results")
		// init container should start and exit with an error, and the regular container should never start
		framework.ExpectNoError(results.Starts(init1))
		framework.ExpectNoError(results.Exits(init1))

		framework.ExpectNoError(results.DoesntStart(regular1))
	})

	ginkgo.It("should run Init container to completion before call to PostStart of regular container", func() {
		init1 := "init-1"
		regular1 := "regular-1"

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "initcontainer-test-pod-with-post-start",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:  init1,
						Image: busyboxImage,
						Command: ExecCommand(init1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							// Allocate sufficient time for its postStart hook
							// to complete.
							// Note that we've observed approximately a 2s
							// delay before the postStart hook is called.
							// 10s > 1s + 2s(estimated maximum delay) + other possible delays
							Delay:    10,
							ExitCode: 0,
						}),
						Lifecycle: &v1.Lifecycle{
							PostStart: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: ExecCommand(prefixedName(PostStartPrefix, regular1), execCommand{
										Delay:         1,
										ExitCode:      0,
										ContainerName: regular1,
									}),
								},
							},
						},
					},
				},
			},
		}

		preparePod(podSpec)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to finish")
		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 1*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, podSpec)

		ginkgo.By("Analyzing results")
		// init container should start and exit with an error, and the regular container should never start
		framework.ExpectNoError(results.StartsBefore(init1, prefixedName(PostStartPrefix, regular1)))
		framework.ExpectNoError(results.ExitsBefore(init1, prefixedName(PostStartPrefix, regular1)))
	})

	ginkgo.It("should restart failing container when pod restartPolicy is Always", func() {

		regular1 := "regular-1"

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "container-must-be-restarted",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    1,
							ExitCode: 1,
						}),
					},
				},
			},
		}

		preparePod(podSpec)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod, it will not finish")
		err := WaitForPodContainerRestartCount(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, 0, 3, 2*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, podSpec)

		ginkgo.By("Analyzing results")
		// container must be restarted
		framework.ExpectNoError(results.Starts(regular1))
		framework.ExpectNoError(results.StartsBefore(regular1, regular1))
		framework.ExpectNoError(results.ExitsBefore(regular1, regular1))
	})

	ginkgo.It("should not launch second container before PostStart of the first container completed", func() {

		regular1 := "regular-1"
		regular2 := "regular-2"

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "post-start-blocks-second-container",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							// Allocate sufficient time for its postStart hook
							// to complete.
							// Note that we've observed approximately a 2s
							// delay before the postStart hook is called.
							// 10s > 1s + 2s(estimated maximum delay) + other possible delays
							Delay:    10,
							ExitCode: 0,
						}),
						Lifecycle: &v1.Lifecycle{
							PostStart: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: ExecCommand(prefixedName(PostStartPrefix, regular1), execCommand{
										Delay:         1,
										ExitCode:      0,
										ContainerName: regular1,
									}),
								},
							},
						},
					},
					{
						Name:  regular2,
						Image: busyboxImage,
						Command: ExecCommand(regular2, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(podSpec)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to finish")
		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 1*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, podSpec)

		ginkgo.By("Analyzing results")
		// second container should not start before the PostStart of a first container completed
		framework.ExpectNoError(results.StartsBefore(prefixedName(PostStartPrefix, regular1), regular2))
		framework.ExpectNoError(results.ExitsBefore(prefixedName(PostStartPrefix, regular1), regular2))
	})

	ginkgo.When("have init container in a Pod with restartPolicy=Never", func() {

		ginkgo.When("an init container fails to start because of a bad image", ginkgo.Ordered, func() {

			init1 := "init1-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "bad-image",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:  init1,
							Image: imageutils.GetE2EImage(imageutils.InvalidRegistryImage),
							Command: ExecCommand(init1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    1,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should mark a Pod as failed and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := WaitForPodInitContainerToFail(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, 0, "ImagePullBackOff", f.Timeouts.PodStart)
				framework.ExpectNoError(err)

				podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should not start an init container", func() {
				framework.ExpectNoError(results.DoesntStart(init1))
			})
			ginkgo.It("should not start a regular container", func() {
				framework.ExpectNoError(results.DoesntStart(regular1))
			})
		})
	})

	ginkgo.It("shouldn't restart init containers upon regular container restart", func() {
		init1 := "init-1"
		init2 := "init-2"
		init3 := "init-3"
		regular1 := "regular-1"

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "initcontainer-test-pod",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				InitContainers: []v1.Container{
					{
						Name:  init1,
						Image: busyboxImage,
						Command: ExecCommand(init1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
					{
						Name:  init2,
						Image: busyboxImage,
						Command: ExecCommand(init2, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
					{
						Name:  init3,
						Image: busyboxImage,
						Command: ExecCommand(init3, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    10,
							ExitCode: -1,
						}),
					},
				},
			},
		}

		preparePod(podSpec)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)
		ginkgo.By("Waiting for the pod to restart a few times")
		err := WaitForPodContainerRestartCount(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, 0, 3, 2*time.Minute)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, podSpec)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.StartsBefore(init1, init2))
		framework.ExpectNoError(results.ExitsBefore(init1, init2))

		framework.ExpectNoError(results.StartsBefore(init2, init3))
		framework.ExpectNoError(results.ExitsBefore(init2, init3))

		framework.ExpectNoError(results.StartsBefore(init3, regular1))
		framework.ExpectNoError(results.ExitsBefore(init3, regular1))

		// ensure that the init containers never restarted
		framework.ExpectNoError(results.HasNotRestarted(init1))
		framework.ExpectNoError(results.HasNotRestarted(init2))
		framework.ExpectNoError(results.HasNotRestarted(init3))
		// while the regular container did
		framework.ExpectNoError(results.HasRestarted(regular1))
	})

	ginkgo.When("a pod cannot terminate gracefully", func() {
		testPod := func(name string, gracePeriod int64) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "busybox",
							Image: imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{
								"sleep",
								"10000",
							},
						},
					},
					TerminationGracePeriodSeconds: &gracePeriod,
				},
			}
		}

		// To account for the time it takes to delete the pod, we add a buffer. Its sized
		// so that we allow up to 2x the grace time to delete the pod. Its extra large to
		// reduce test flakes.
		bufferSeconds := int64(30)

		f.It("should respect termination grace period seconds", f.WithNodeConformance(), func() {
			client := e2epod.NewPodClient(f)
			gracePeriod := int64(30)

			ginkgo.By("creating a pod with a termination grace period seconds")
			pod := testPod("pod-termination-grace-period", gracePeriod)
			pod = client.Create(context.TODO(), pod)

			ginkgo.By("ensuring the pod is running")
			err := e2epod.WaitForPodRunningInNamespace(context.TODO(), f.ClientSet, pod)
			framework.ExpectNoError(err)

			ginkgo.By("deleting the pod gracefully")
			err = client.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("ensuring the pod is terminated within the grace period seconds + buffer seconds")
			err = e2epod.WaitForPodNotFoundInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, time.Duration(gracePeriod+bufferSeconds)*time.Second)
			framework.ExpectNoError(err)
		})

		f.It("should respect termination grace period seconds with long-running preStop hook", f.WithNodeConformance(), func() {
			client := e2epod.NewPodClient(f)
			gracePeriod := int64(30)

			ginkgo.By("creating a pod with a termination grace period seconds and long-running preStop hook")
			pod := testPod("pod-termination-grace-period", gracePeriod)
			pod.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: []string{
							"sleep",
							"10000",
						},
					},
				},
			}
			pod = client.Create(context.TODO(), pod)

			ginkgo.By("ensuring the pod is running")
			err := e2epod.WaitForPodRunningInNamespace(context.TODO(), f.ClientSet, pod)
			framework.ExpectNoError(err)

			ginkgo.By("deleting the pod gracefully")
			err = client.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("ensuring the pod is terminated within the grace period seconds + buffer seconds")
			err = e2epod.WaitForPodNotFoundInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, time.Duration(gracePeriod+bufferSeconds)*time.Second)
			framework.ExpectNoError(err)
		})
	})

	ginkgo.It("should call the container's preStop hook and terminate it if its startup probe fails", func() {
		regular1 := "regular-1"

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-pod",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:              100,
							TerminationSeconds: 15,
							ExitCode:           0,
						}),
						StartupProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{
										"sh",
										"-c",
										"exit 1",
									},
								},
							},
							InitialDelaySeconds: 10,
							FailureThreshold:    1,
						},
						Lifecycle: &v1.Lifecycle{
							PreStop: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: ExecCommand(prefixedName(PreStopPrefix, regular1), execCommand{
										Delay:         1,
										ExitCode:      0,
										ContainerName: regular1,
									}),
								},
							},
						},
					},
				},
			},
		}

		preparePod(podSpec)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)

		ginkgo.By("Waiting for the pod to complete")
		err := e2epod.WaitForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, podSpec)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.RunTogether(regular1, prefixedName(PreStopPrefix, regular1)))
		framework.ExpectNoError(results.Starts(prefixedName(PreStopPrefix, regular1)))
		framework.ExpectNoError(results.Exits(regular1))
	})

	ginkgo.It("should call the container's preStop hook and terminate it if its liveness probe fails", func() {
		regular1 := "regular-1"

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-pod",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:              100,
							TerminationSeconds: 15,
							ExitCode:           0,
						}),
						LivenessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{
										"sh",
										"-c",
										"exit 1",
									},
								},
							},
							InitialDelaySeconds: 10,
							FailureThreshold:    1,
						},
						Lifecycle: &v1.Lifecycle{
							PreStop: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: ExecCommand(prefixedName(PreStopPrefix, regular1), execCommand{
										Delay:         1,
										ExitCode:      0,
										ContainerName: regular1,
									}),
								},
							},
						},
					},
				},
			},
		}

		preparePod(podSpec)

		client := e2epod.NewPodClient(f)
		podSpec = client.Create(context.TODO(), podSpec)

		ginkgo.By("Waiting for the pod to complete")
		err := e2epod.WaitForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace)
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		podSpec, err = client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, podSpec)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.RunTogether(regular1, prefixedName(PreStopPrefix, regular1)))
		framework.ExpectNoError(results.Starts(prefixedName(PreStopPrefix, regular1)))
		framework.ExpectNoError(results.Exits(regular1))
	})

	ginkgo.When("a pod is terminating because its liveness probe fails", func() {
		regular1 := "regular-1"

		testPod := func() *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: v1.PodSpec{
					RestartPolicy:                 v1.RestartPolicyNever,
					TerminationGracePeriodSeconds: ptr.To(int64(100)),
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: imageutils.GetE2EImage(imageutils.BusyBox),
							Command: ExecCommand(regular1, execCommand{
								Delay:              100,
								TerminationSeconds: 15,
								ExitCode:           0,
							}),
							LivenessProbe: &v1.Probe{
								ProbeHandler: v1.ProbeHandler{
									Exec: &v1.ExecAction{
										Command: ExecCommand(prefixedName(LivenessPrefix, regular1), execCommand{
											ExitCode:      1,
											ContainerName: regular1,
										}),
									},
								},
								InitialDelaySeconds: 10,
								PeriodSeconds:       1,
								FailureThreshold:    1,
							},
						},
					},
				},
			}
		}

		f.It("should execute readiness probe while in preStop, but not liveness", f.WithNodeConformance(), func() {
			client := e2epod.NewPodClient(f)
			podSpec := testPod()

			ginkgo.By("creating a pod with a readiness probe and a preStop hook")
			podSpec.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: ExecCommand(prefixedName(PreStopPrefix, regular1), execCommand{
							Delay:         1,
							ExitCode:      0,
							ContainerName: regular1,
						}),
					},
				},
			}
			podSpec.Spec.Containers[0].ReadinessProbe = &v1.Probe{
				ProbeHandler: v1.ProbeHandler{
					Exec: &v1.ExecAction{
						Command: ExecCommand(prefixedName(ReadinessPrefix, regular1), execCommand{
							ExitCode:      0,
							ContainerName: regular1,
						}),
					},
				},
				InitialDelaySeconds: 1,
				PeriodSeconds:       1,
			}

			preparePod(podSpec)

			podSpec = client.Create(context.TODO(), podSpec)

			ginkgo.By("Waiting for the pod to complete")
			err := e2epod.WaitForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace)
			framework.ExpectNoError(err)

			ginkgo.By("Parsing results")
			podSpec, err = client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			results := parseOutput(context.TODO(), f, podSpec)

			ginkgo.By("Analyzing results")
			// readiness probes are called during pod termination
			framework.ExpectNoError(results.RunTogether(prefixedName(PreStopPrefix, regular1), prefixedName(ReadinessPrefix, regular1)))
			// liveness probes are not called during pod termination
			err = results.RunTogether(prefixedName(PreStopPrefix, regular1), prefixedName(LivenessPrefix, regular1))
			gomega.Expect(err).To(gomega.HaveOccurred())
		})

		f.It("should continue running liveness probes for restartable init containers and restart them while in preStop", f.WithNodeConformance(), func() {
			client := e2epod.NewPodClient(f)
			podSpec := testPod()
			restartableInit1 := "restartable-init-1"

			ginkgo.By("creating a pod with a restartable init container and a preStop hook")
			podSpec.Spec.InitContainers = []v1.Container{{
				RestartPolicy: &containerRestartPolicyAlways,
				Name:          restartableInit1,
				Image:         imageutils.GetE2EImage(imageutils.BusyBox),
				Command: ExecCommand(restartableInit1, execCommand{
					Delay:              100,
					TerminationSeconds: 1,
					ExitCode:           0,
				}),
				LivenessProbe: &v1.Probe{
					ProbeHandler: v1.ProbeHandler{
						Exec: &v1.ExecAction{
							Command: ExecCommand(prefixedName(LivenessPrefix, restartableInit1), execCommand{
								ExitCode:      1,
								ContainerName: restartableInit1,
							}),
						},
					},
					InitialDelaySeconds: 1,
					PeriodSeconds:       1,
					FailureThreshold:    1,
				},
			}}
			podSpec.Spec.Containers[0].Lifecycle = &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: ExecCommand(prefixedName(PreStopPrefix, regular1), execCommand{
							Delay:         40,
							ExitCode:      0,
							ContainerName: regular1,
						}),
					},
				},
			}

			preparePod(podSpec)

			podSpec = client.Create(context.TODO(), podSpec)

			ginkgo.By("Waiting for the pod to complete")
			err := e2epod.WaitForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace)
			framework.ExpectNoError(err)

			ginkgo.By("Parsing results")
			podSpec, err = client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			results := parseOutput(context.TODO(), f, podSpec)

			ginkgo.By("Analyzing results")
			// FIXME ExpectNoError: this will be implemented in KEP 4438
			// liveness probes are called for restartable init containers during pod termination
			err = results.RunTogether(prefixedName(PreStopPrefix, regular1), prefixedName(LivenessPrefix, restartableInit1))
			gomega.Expect(err).To(gomega.HaveOccurred())
			// FIXME ExpectNoError: this will be implemented in KEP 4438
			// restartable init containers are restarted during pod termination
			err = results.RunTogether(prefixedName(PreStopPrefix, regular1), restartableInit1)
			gomega.Expect(err).To(gomega.HaveOccurred())
		})
	})
})

var _ = SIGDescribe(framework.WithSerial(), "Containers Lifecycle", func() {
	f := framework.NewDefaultFramework("containers-lifecycle-test-serial")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should restart the containers in right order after the node reboot", func(ctx context.Context) {
		init1 := "init-1"
		init2 := "init-2"
		init3 := "init-3"
		regular1 := "regular-1"

		podLabels := map[string]string{
			"test":      "containers-lifecycle-test-serial",
			"namespace": f.Namespace.Name,
		}
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "initialized-pod",
				Labels: podLabels,
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				InitContainers: []v1.Container{
					{
						Name:  init1,
						Image: busyboxImage,
						Command: ExecCommand(init1, execCommand{
							Delay:    5,
							ExitCode: 0,
						}),
					},
					{
						Name:  init2,
						Image: busyboxImage,
						Command: ExecCommand(init2, execCommand{
							Delay:    5,
							ExitCode: 0,
						}),
					},
					{
						Name:  init3,
						Image: busyboxImage,
						Command: ExecCommand(init3, execCommand{
							Delay:    5,
							ExitCode: 0,
						}),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    30,
							ExitCode: 0,
						}),
					},
				},
			},
		}
		preparePod(pod)

		client := e2epod.NewPodClient(f)
		pod = client.Create(ctx, pod)
		ginkgo.By("Waiting for the pod to be initialized and run")
		err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
		framework.ExpectNoError(err)

		ginkgo.By("Getting the current pod sandbox ID")
		rs, _, err := getCRIClient()
		framework.ExpectNoError(err)

		sandboxes, err := rs.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{
			LabelSelector: podLabels,
		})
		framework.ExpectNoError(err)
		gomega.Expect(sandboxes).To(gomega.HaveLen(1))
		podSandboxID := sandboxes[0].Id

		ginkgo.By("Stopping the kubelet")
		restartKubelet := stopKubelet()
		gomega.Eventually(ctx, func() bool {
			return kubeletHealthCheck(kubeletHealthCheckURL)
		}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeFalse())

		ginkgo.By("Stopping the pod sandbox to simulate the node reboot")
		err = rs.StopPodSandbox(ctx, podSandboxID)
		framework.ExpectNoError(err)

		ginkgo.By("Restarting the kubelet")
		restartKubelet()
		gomega.Eventually(ctx, func() bool {
			return kubeletHealthCheck(kubeletHealthCheckURL)
		}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrue())

		ginkgo.By("Waiting for the pod to be re-initialized and run")
		err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "re-initialized", f.Timeouts.PodStart, func(pod *v1.Pod) (bool, error) {
			if pod.Status.ContainerStatuses[0].RestartCount < 2 {
				return false, nil
			}
			if pod.Status.Phase != v1.PodRunning {
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		pod, err = client.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, pod)

		ginkgo.By("Analyzing results")
		init1Started, err := results.FindIndex(init1, "Started", 0)
		framework.ExpectNoError(err)
		init2Started, err := results.FindIndex(init2, "Started", 0)
		framework.ExpectNoError(err)
		init3Started, err := results.FindIndex(init3, "Started", 0)
		framework.ExpectNoError(err)
		regular1Started, err := results.FindIndex(regular1, "Started", 0)
		framework.ExpectNoError(err)

		init1Restarted, err := results.FindIndex(init1, "Started", init1Started+1)
		framework.ExpectNoError(err)
		init2Restarted, err := results.FindIndex(init2, "Started", init2Started+1)
		framework.ExpectNoError(err)
		init3Restarted, err := results.FindIndex(init3, "Started", init3Started+1)
		framework.ExpectNoError(err)
		regular1Restarted, err := results.FindIndex(regular1, "Started", regular1Started+1)
		framework.ExpectNoError(err)

		framework.ExpectNoError(init1Started.IsBefore(init2Started))
		framework.ExpectNoError(init2Started.IsBefore(init3Started))
		framework.ExpectNoError(init3Started.IsBefore(regular1Started))

		framework.ExpectNoError(init1Restarted.IsBefore(init2Restarted))
		framework.ExpectNoError(init2Restarted.IsBefore(init3Restarted))
		framework.ExpectNoError(init3Restarted.IsBefore(regular1Restarted))
	})
})

var _ = SIGDescribe(nodefeature.SidecarContainers, "Containers Lifecycle", func() {
	f := framework.NewDefaultFramework("containers-lifecycle-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.When("using a Pod with restartPolicy=Never, three init container and two restartable init containers", ginkgo.Ordered, func() {

		init1 := "init-1"
		restartableInit1 := "restartable-init-1"
		init2 := "init-2"
		restartableInit2 := "restartable-init-2"
		init3 := "init-3"
		regular1 := "regular-1"

		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "restartable-init-containers-start-serially",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:  init1,
						Image: busyboxImage,
						Command: ExecCommand(init1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
					{
						Name:  restartableInit1,
						Image: busyboxImage,
						Command: ExecCommand(restartableInit1, execCommand{
							Delay:    600,
							ExitCode: 0,
						}),
						RestartPolicy: &containerRestartPolicyAlways,
					},
					{
						Name:  init2,
						Image: busyboxImage,
						Command: ExecCommand(init2, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
					{
						Name:  restartableInit2,
						Image: busyboxImage,
						Command: ExecCommand(restartableInit2, execCommand{
							Delay:    600,
							ExitCode: 0,
						}),
						RestartPolicy: &containerRestartPolicyAlways,
					},
					{
						Name:  init3,
						Image: busyboxImage,
						Command: ExecCommand(init3, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(podSpec)
		var results containerOutputList

		ginkgo.It("should finish and produce log", func() {
			client := e2epod.NewPodClient(f)
			podSpec = client.Create(context.TODO(), podSpec)

			err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 5*time.Minute)
			framework.ExpectNoError(err)

			podSpec, err := client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// pod should exit successfully
			gomega.Expect(podSpec.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

			results = parseOutput(context.TODO(), f, podSpec)
		})

		ginkgo.It("should run the first init container to completion before starting first restartable init container", func() {
			framework.ExpectNoError(results.StartsBefore(init1, restartableInit1))
			framework.ExpectNoError(results.ExitsBefore(init1, restartableInit1))
		})

		ginkgo.It("should start first restartable init container before starting second init container", func() {
			framework.ExpectNoError(results.StartsBefore(restartableInit1, init2))
		})

		ginkgo.It("should run first init container and first restartable init container together", func() {
			framework.ExpectNoError(results.RunTogether(restartableInit1, init2))
		})

		ginkgo.It("should run second init container to completion before starting second restartable init container", func() {
			framework.ExpectNoError(results.StartsBefore(init2, restartableInit2))
			framework.ExpectNoError(results.ExitsBefore(init2, restartableInit2))
		})

		ginkgo.It("should start second restartable init container before third init container", func() {
			framework.ExpectNoError(results.StartsBefore(restartableInit2, init3))
		})

		ginkgo.It("should run both restartable init containers and third init container together", func() {
			framework.ExpectNoError(results.RunTogether(restartableInit1, restartableInit2))
			framework.ExpectNoError(results.RunTogether(restartableInit1, init3))
			framework.ExpectNoError(results.RunTogether(restartableInit2, init3))
		})

		ginkgo.It("should run third init container to completion before starting regular container", func() {
			framework.ExpectNoError(results.StartsBefore(init3, regular1))
			framework.ExpectNoError(results.ExitsBefore(init3, regular1))
		})

		ginkgo.It("should run both restartable init containers and a regular container together", func() {
			framework.ExpectNoError(results.RunTogether(restartableInit1, regular1))
			framework.ExpectNoError(results.RunTogether(restartableInit2, regular1))
		})
	})

	ginkgo.When("using a restartable init container in a Pod with restartPolicy=Never", func() {
		ginkgo.When("a restartable init container runs continuously", ginkgo.Ordered, func() {

			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-run-continuously",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    1,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should complete a Pod successfully and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 5*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				// pod should exit successfully
				gomega.Expect(podSpec.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should not restart a restartable init container", func() {
				framework.ExpectNoError(results.DoesntStartAfter(restartableInit1, regular1))
			})
			ginkgo.It("should run a regular container to completion", func() {
				framework.ExpectNoError(results.Exits(regular1))
			})
		})

		ginkgo.When("a restartable init container fails to start because of a bad image", ginkgo.Ordered, func() {

			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-runs-with-pod",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: imageutils.GetE2EImage(imageutils.InvalidRegistryImage),
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    1,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should mark a Pod as failed and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				// restartable init container should be in image pull backoff
				err := WaitForPodInitContainerToFail(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, 0, "ImagePullBackOff", f.Timeouts.PodStart)
				framework.ExpectNoError(err)

				podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should not start a restartable init container", func() {
				framework.ExpectNoError(results.DoesntStart(restartableInit1))
			})
			ginkgo.It("should not start a regular container", func() {
				framework.ExpectNoError(results.DoesntStart(regular1))
			})
		})

		// TODO: add a test case similar to one above, but with startup probe never succeeding

		ginkgo.When("a restartable init container starts and exits with exit code 0 continuously", ginkgo.Ordered, func() {

			restartableInit1 := "restartable-init-1"
			init1 := "init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-exit-0-continuously",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    5,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    5,
								ExitCode: 0,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    60,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should complete a Pod successfully and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 5*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				// pod should exit successfully
				gomega.Expect(podSpec.Status.Phase).To(gomega.Equal(v1.PodSucceeded))
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should restart a restartable init container before the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(restartableInit1, regular1))
			})
			ginkgo.It("should restart a restartable init container after the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(regular1, restartableInit1))
			})
			ginkgo.It("should run a regular container to completion", func() {
				framework.ExpectNoError(results.Exits(regular1))
			})
		})

		ginkgo.When("a restartable init container starts and exits with exit code 1 continuously", ginkgo.Ordered, func() {
			restartableInit1 := "restartable-init-1"
			init1 := "init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-exit-1-continuously",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    5,
								ExitCode: 1,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    5,
								ExitCode: 0,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    60,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should complete a Pod successfully and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 5*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				// pod should exit successfully
				gomega.Expect(podSpec.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should restart a restartable init container before the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(restartableInit1, regular1))
			})
			ginkgo.It("should restart a restartable init container after the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(regular1, restartableInit1))
			})
			ginkgo.It("should run a regular container to completion", func() {
				framework.ExpectNoError(results.Exits(regular1))
			})
		})

		ginkgo.When("an Init container before restartable init container fails", ginkgo.Ordered, func() {

			init1 := "init-1"
			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "init-container-fails-before-restartable-init-starts",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    1,
								ExitCode: 1,
							}),
						},
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should mark a Pod as failed and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitForPodFailedReason(context.TODO(), f.ClientSet, podSpec, "", 1*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should mark an Init container as failed", func() {
				framework.ExpectNoError(results.Exits(init1))
			})
			ginkgo.It("should not start restartable init container", func() {
				framework.ExpectNoError(results.DoesntStart(restartableInit1))
			})
		})

		ginkgo.When("an Init container after restartable init container fails", ginkgo.Ordered, func() {

			init1 := "init-1"
			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-fails-before-init-container",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    5,
								ExitCode: 1,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    1,
								ExitCode: 1,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should mark a Pod as failed and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitForPodFailedReason(context.TODO(), f.ClientSet, podSpec, "", 1*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should mark an Init container as failed", func() {
				framework.ExpectNoError(results.Exits(init1))
			})
			// TODO: how will we be able to test it if restartable init container
			// will never fail and there will be no termination log? Or will be?
			ginkgo.It("should be running restartable init container and a failed Init container in parallel", func() {
				framework.ExpectNoError(results.RunTogether(restartableInit1, init1))
			})
		})
	})

	ginkgo.When("using a restartable init container in a Pod with restartPolicy=OnFailure", ginkgo.Ordered, func() {
		// this test case the same as for restartPolicy=Never
		ginkgo.When("a restartable init container runs continuously", func() {

			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-run-continuously",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    1,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should complete a Pod successfully and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 5*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				// pod should exit successfully
				gomega.Expect(podSpec.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should not restart a restartable init container", func() {
				framework.ExpectNoError(results.DoesntStartAfter(restartableInit1, regular1))
			})
			ginkgo.It("should run a regular container to completion", func() {
				framework.ExpectNoError(results.Exits(regular1))
			})
		})

		ginkgo.When("a restartable init container fails to start because of a bad image", ginkgo.Ordered, func() {

			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-runs-with-pod",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: imageutils.GetE2EImage(imageutils.InvalidRegistryImage),
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    1,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should mark a Pod as failed and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				// restartable init container should be in image pull backoff
				err := WaitForPodInitContainerToFail(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, 0, "ImagePullBackOff", f.Timeouts.PodStart)
				framework.ExpectNoError(err)

				podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should not start a restartable init container", func() {
				framework.ExpectNoError(results.DoesntStart(restartableInit1))
			})
			ginkgo.It("should not start a regular container", func() {
				framework.ExpectNoError(results.DoesntStart(regular1))
			})
		})

		// TODO: add a test case similar to one above, but with startup probe never succeeding

		// this test case the same as for restartPolicy=Never
		ginkgo.When("a restartable init container starts and exits with exit code 0 continuously", ginkgo.Ordered, func() {

			restartableInit1 := "restartable-init-1"
			init1 := "init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-exit-0-continuously",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    5,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    5,
								ExitCode: 0,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    60,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should complete a Pod successfully and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 5*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				// pod should exit successfully
				gomega.Expect(podSpec.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should restart a restartable init container before the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(restartableInit1, regular1))
			})
			ginkgo.It("should restart a restartable init container after the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(regular1, restartableInit1))
			})
			ginkgo.It("should run a regular container to completion", func() {
				framework.ExpectNoError(results.Exits(regular1))
			})
		})

		// this test case the same as for restartPolicy=Never
		ginkgo.When("a restartable init container starts and exits with exit code 1 continuously", ginkgo.Ordered, func() {

			restartableInit1 := "restartable-init-1"
			init1 := "init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-exit-1-continuously",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    5,
								ExitCode: 1,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    5,
								ExitCode: 0,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    60,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should complete a Pod successfully and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 5*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				// pod should exit successfully
				gomega.Expect(podSpec.Status.Phase).To(gomega.Equal(v1.PodSucceeded))

				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should restart a restartable init container before the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(restartableInit1, regular1))
			})
			ginkgo.It("should restart a restartable init container after the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(regular1, restartableInit1))
			})
			ginkgo.It("should run a regular container to completion", func() {
				framework.ExpectNoError(results.Exits(regular1))
			})
		})

		ginkgo.When("an Init container before restartable init container continuously fails", ginkgo.Ordered, func() {

			init1 := "init-1"
			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "init-container-fails-before-restartable-init-starts",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
					InitContainers: []v1.Container{
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    1,
								ExitCode: 1,
							}),
						},
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should continuously run Pod keeping it Pending", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitForPodCondition(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, "pending and restarting 3 times", 5*time.Minute, func(pod *v1.Pod) (bool, error) {
					if pod.Status.Phase != v1.PodPending {
						return false, fmt.Errorf("pod should be in pending phase")
					}
					if len(pod.Status.InitContainerStatuses) < 1 {
						return false, nil
					}
					containerStatus := pod.Status.InitContainerStatuses[0]
					return containerStatus.RestartCount >= 3, nil
				})
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should have Init container restartCount greater than 0", func() {
				framework.ExpectNoError(results.HasRestarted(init1))
			})
			ginkgo.It("should not start restartable init container", func() {
				framework.ExpectNoError(results.DoesntStart(restartableInit1))
			})
		})

		ginkgo.When("an Init container after restartable init container fails", ginkgo.Ordered, func() {

			init1 := "init-1"
			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-fails-before-init-container",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyOnFailure,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    5,
								ExitCode: 1,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    1,
								ExitCode: 1,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should continuously run Pod keeping it Pending", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitForPodCondition(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, "pending and restarting 3 times", 5*time.Minute, func(pod *v1.Pod) (bool, error) {
					if pod.Status.Phase != v1.PodPending {
						return false, fmt.Errorf("pod should be in pending phase")
					}
					if len(pod.Status.InitContainerStatuses) < 1 {
						return false, nil
					}
					containerStatus := pod.Status.InitContainerStatuses[0]
					return containerStatus.RestartCount >= 3, nil
				})
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should have Init container restartCount greater than 0", func() {
				framework.ExpectNoError(results.HasRestarted(init1))
			})
			// TODO: how will we be able to test it if restartable init container will never fail and there will be no termination log? Or will be?
			ginkgo.It("should be running restartable init container and a failed Init container in parallel", func() {
				framework.ExpectNoError(results.RunTogether(restartableInit1, init1))
			})
		})
	})

	ginkgo.When("using a restartable init container in a Pod with restartPolicy=Always", ginkgo.Ordered, func() {
		ginkgo.When("a restartable init container runs continuously", func() {

			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-run-continuously",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    1,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			// regular container should exit at least once so we can get it's termination log
			// this test case is different from restartPolicy=Never
			ginkgo.It("should keep running a Pod continuously and produce log", func() { /* check the regular container restartCount > 0 */
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := WaitForPodContainerRestartCount(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, 0, 2, 2*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})

			ginkgo.It("should not restart a restartable init container", func() {
				framework.ExpectNoError(results.DoesntStartAfter(restartableInit1, regular1))
			})
			// this test case is different from restartPolicy=Never
			ginkgo.It("should start a regular container", func() {
				framework.ExpectNoError(results.HasRestarted(regular1))
			})
		})

		ginkgo.When("a restartable init container fails to start because of a bad image", ginkgo.Ordered, func() {

			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-runs-with-pod",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: imageutils.GetE2EImage(imageutils.InvalidRegistryImage),
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    1,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should continuously run Pod keeping it Pending and produce log", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				// restartable init container should be in image pull backoff
				err := WaitForPodInitContainerToFail(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, 0, "ImagePullBackOff", f.Timeouts.PodStart)
				framework.ExpectNoError(err)

				podSpec, err = client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should not start a restartable init container", func() {
				framework.ExpectNoError(results.DoesntStart(restartableInit1))
			})
			ginkgo.It("should not start a regular container", func() {
				framework.ExpectNoError(results.DoesntStart(regular1))
			})
		})

		// TODO: add a test case similar to one above, but with startup probe never succeeding

		ginkgo.When("a restartable init container starts and exits with exit code 0 continuously", ginkgo.Ordered, func() {

			restartableInit1 := "restartable-init-1"
			init1 := "init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-exit-0-continuously",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    5,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    5,
								ExitCode: 0,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    60,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should keep running a Pod continuously and produce log", func() { /* check the regular container restartCount > 0 */
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := WaitForPodContainerRestartCount(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, 0, 1, 2*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should restart a restartable init container before the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(restartableInit1, regular1))
			})
			ginkgo.It("should restart a restartable init container after the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(regular1, restartableInit1))
			})
			ginkgo.It("should start a regular container", func() {
				framework.ExpectNoError(results.Starts(regular1))
			})
		})

		// this test case the same as for restartPolicy=Never
		ginkgo.When("a restartable init container starts and exits with exit code 1 continuously", ginkgo.Ordered, func() {

			restartableInit1 := "restartable-init-1"
			init1 := "init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-exit-1-continuously",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    5,
								ExitCode: 1,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    5,
								ExitCode: 0,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    60,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should keep running a Pod continuously and produce log", func() { /* check the regular container restartCount > 0 */
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := WaitForPodContainerRestartCount(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, 0, 1, 2*time.Minute)
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should restart a restartable init container before the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(restartableInit1, regular1))
			})
			ginkgo.It("should restart a restartable init container after the regular container started", func() {
				framework.ExpectNoError(results.StartsBefore(regular1, restartableInit1))
			})
			ginkgo.It("should start a regular container", func() {
				framework.ExpectNoError(results.Starts(regular1))
			})
		})

		ginkgo.When("an Init container before restartable init container continuously fails", ginkgo.Ordered, func() {

			init1 := "init-1"
			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "init-container-fails-before-restartable-init-starts",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					InitContainers: []v1.Container{
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    1,
								ExitCode: 1,
							}),
						},
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should continuously run Pod keeping it Pending", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitForPodCondition(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, "pending and restarting 3 times", 5*time.Minute, func(pod *v1.Pod) (bool, error) {
					if pod.Status.Phase != v1.PodPending {
						return false, fmt.Errorf("pod should be in pending phase")
					}
					if len(pod.Status.InitContainerStatuses) < 1 {
						return false, nil
					}
					containerStatus := pod.Status.InitContainerStatuses[0]
					return containerStatus.RestartCount >= 3, nil
				})
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should have Init container restartCount greater than 0", func() {
				framework.ExpectNoError(results.HasRestarted(init1))
			})
			ginkgo.It("should not start restartable init container", func() {
				framework.ExpectNoError(results.DoesntStart(restartableInit1))
			})
		})

		ginkgo.When("an Init container after restartable init container fails", ginkgo.Ordered, func() {

			init1 := "init-1"
			restartableInit1 := "restartable-init-1"
			regular1 := "regular-1"

			podSpec := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "restartable-init-container-fails-before-init-container",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					InitContainers: []v1.Container{
						{
							Name:  restartableInit1,
							Image: busyboxImage,
							Command: ExecCommand(restartableInit1, execCommand{
								Delay:    5,
								ExitCode: 1,
							}),
							RestartPolicy: &containerRestartPolicyAlways,
						},
						{
							Name:  init1,
							Image: busyboxImage,
							Command: ExecCommand(init1, execCommand{
								Delay:    1,
								ExitCode: 1,
							}),
						},
					},
					Containers: []v1.Container{
						{
							Name:  regular1,
							Image: busyboxImage,
							Command: ExecCommand(regular1, execCommand{
								Delay:    600,
								ExitCode: 0,
							}),
						},
					},
				},
			}

			preparePod(podSpec)
			var results containerOutputList

			ginkgo.It("should continuously run Pod keeping it Pending", func() {
				client := e2epod.NewPodClient(f)
				podSpec = client.Create(context.TODO(), podSpec)

				err := e2epod.WaitForPodCondition(context.TODO(), f.ClientSet, podSpec.Namespace, podSpec.Name, "pending and restarting 3 times", 5*time.Minute, func(pod *v1.Pod) (bool, error) {
					if pod.Status.Phase != v1.PodPending {
						return false, fmt.Errorf("pod should be in pending phase")
					}
					if len(pod.Status.InitContainerStatuses) < 1 {
						return false, nil
					}
					containerStatus := pod.Status.InitContainerStatuses[0]
					return containerStatus.RestartCount >= 3, nil
				})
				framework.ExpectNoError(err)

				podSpec, err := client.Get(context.TODO(), podSpec.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				results = parseOutput(context.TODO(), f, podSpec)
			})
			ginkgo.It("should have Init container restartCount greater than 0", func() {
				framework.ExpectNoError(results.HasRestarted(init1))
			})
			// TODO: how will we be able to test it if restartable init container will never fail and there will be no termination log? Or will be?
			ginkgo.It("should be running restartable init container and a failed Init container in parallel", func() {
				framework.ExpectNoError(results.RunTogether(restartableInit1, init1))
			})
		})
	})

	ginkgo.It("should launch restartable init containers serially considering the startup probe", func() {

		restartableInit1 := "restartable-init-1"
		restartableInit2 := "restartable-init-2"
		regular1 := "regular-1"

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "restartable-init-containers-start-serially",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:  restartableInit1,
						Image: busyboxImage,
						Command: ExecCommand(restartableInit1, execCommand{
							StartDelay: 10,
							Delay:      600,
							ExitCode:   0,
						}),
						StartupProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{"test", "-f", "started"},
								},
							},
						},
						RestartPolicy: &containerRestartPolicyAlways,
					},
					{
						Name:  restartableInit2,
						Image: busyboxImage,
						Command: ExecCommand(restartableInit2, execCommand{
							StartDelay: 10,
							Delay:      600,
							ExitCode:   0,
						}),
						StartupProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{"test", "-f", "started"},
								},
							},
						},
						RestartPolicy: &containerRestartPolicyAlways,
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(pod)

		client := e2epod.NewPodClient(f)
		pod = client.Create(context.TODO(), pod)

		ginkgo.By("Waiting for the pod to finish")
		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, 5*time.Minute)
		framework.ExpectNoError(err)

		pod, err = client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, pod)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.StartsBefore(restartableInit1, restartableInit2))
		framework.ExpectNoError(results.StartsBefore(restartableInit2, regular1))
	})

	ginkgo.It("should call the container's preStop hook and not launch next container if the restartable init container's startup probe fails", func() {

		restartableInit1 := "restartable-init-1"
		regular1 := "regular-1"

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "restartable-init-container-failed-startup",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				InitContainers: []v1.Container{
					{
						Name:  restartableInit1,
						Image: busyboxImage,
						Command: ExecCommand(restartableInit1, execCommand{
							Delay:              600,
							TerminationSeconds: 15,
							ExitCode:           0,
						}),
						StartupProbe: &v1.Probe{
							InitialDelaySeconds: 5,
							FailureThreshold:    1,
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{
										"sh",
										"-c",
										"exit 1",
									},
								},
							},
						},
						Lifecycle: &v1.Lifecycle{
							PreStop: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: ExecCommand(prefixedName(PreStopPrefix, restartableInit1), execCommand{
										Delay:         1,
										ExitCode:      0,
										ContainerName: restartableInit1,
									}),
								},
							},
						},
						RestartPolicy: &containerRestartPolicyAlways,
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(pod)

		client := e2epod.NewPodClient(f)
		pod = client.Create(context.TODO(), pod)

		ginkgo.By("Waiting for the restartable init container to restart")
		err := WaitForPodInitContainerRestartCount(context.TODO(), f.ClientSet, pod.Namespace, pod.Name, 0, 2, 2*time.Minute)
		framework.ExpectNoError(err)

		pod, err = client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		if pod.Status.Phase != v1.PodPending {
			framework.Failf("pod %q is not pending, it's %q", pod.Name, pod.Status.Phase)
		}

		results := parseOutput(context.TODO(), f, pod)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.RunTogether(restartableInit1, prefixedName(PreStopPrefix, restartableInit1)))
		framework.ExpectNoError(results.Starts(prefixedName(PreStopPrefix, restartableInit1)))
		framework.ExpectNoError(results.Exits(restartableInit1))
		framework.ExpectNoError(results.DoesntStart(regular1))
	})

	ginkgo.It("should call the container's preStop hook and start the next container if the restartable init container's liveness probe fails", func() {

		restartableInit1 := "restartable-init-1"
		regular1 := "regular-1"

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "restartable-init-container-failed-startup",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				InitContainers: []v1.Container{
					{
						Name:  restartableInit1,
						Image: busyboxImage,
						Command: ExecCommand(restartableInit1, execCommand{
							Delay:              600,
							TerminationSeconds: 15,
							ExitCode:           0,
						}),
						LivenessProbe: &v1.Probe{
							InitialDelaySeconds: 5,
							FailureThreshold:    1,
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{
										"sh",
										"-c",
										"exit 1",
									},
								},
							},
						},
						Lifecycle: &v1.Lifecycle{
							PreStop: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: ExecCommand(prefixedName(PreStopPrefix, restartableInit1), execCommand{
										Delay:         1,
										ExitCode:      0,
										ContainerName: restartableInit1,
									}),
								},
							},
						},
						RestartPolicy: &containerRestartPolicyAlways,
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    1,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(pod)

		client := e2epod.NewPodClient(f)
		pod = client.Create(context.TODO(), pod)

		ginkgo.By("Waiting for the restartable init container to restart")
		err := WaitForPodInitContainerRestartCount(context.TODO(), f.ClientSet, pod.Namespace, pod.Name, 0, 2, 2*time.Minute)
		framework.ExpectNoError(err)

		err = WaitForPodContainerRestartCount(context.TODO(), f.ClientSet, pod.Namespace, pod.Name, 0, 1, 2*time.Minute)
		framework.ExpectNoError(err)

		pod, err = client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		results := parseOutput(context.TODO(), f, pod)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.RunTogether(restartableInit1, prefixedName(PreStopPrefix, restartableInit1)))
		framework.ExpectNoError(results.Starts(prefixedName(PreStopPrefix, restartableInit1)))
		framework.ExpectNoError(results.Exits(restartableInit1))
		framework.ExpectNoError(results.Starts(regular1))
	})

	ginkgo.It("should terminate sidecars in reverse order after all main containers have exited", func() {
		restartableInit1 := "restartable-init-1"
		restartableInit2 := "restartable-init-2"
		restartableInit3 := "restartable-init-3"
		regular1 := "regular-1"

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "serialize-termination",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:          restartableInit1,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit1, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
					},
					{
						Name:          restartableInit2,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit2, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
					},
					{
						Name:          restartableInit3,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit3, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    5,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(pod)

		client := e2epod.NewPodClient(f)
		pod = client.Create(context.TODO(), pod)

		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, 5*time.Minute)
		framework.ExpectNoError(err)

		pod, err = client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		results := parseOutput(context.TODO(), f, pod)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.StartsBefore(restartableInit1, restartableInit2))
		framework.ExpectNoError(results.StartsBefore(restartableInit1, restartableInit3))
		framework.ExpectNoError(results.StartsBefore(restartableInit2, restartableInit3))
		framework.ExpectNoError(results.StartsBefore(restartableInit1, regular1))
		framework.ExpectNoError(results.StartsBefore(restartableInit2, regular1))
		framework.ExpectNoError(results.StartsBefore(restartableInit3, regular1))

		// main containers exit first
		framework.ExpectNoError(results.ExitsBefore(regular1, restartableInit1))
		framework.ExpectNoError(results.ExitsBefore(regular1, restartableInit2))
		framework.ExpectNoError(results.ExitsBefore(regular1, restartableInit3))
		// followed by sidecars in reverse order
		framework.ExpectNoError(results.ExitsBefore(restartableInit3, restartableInit2))
		framework.ExpectNoError(results.ExitsBefore(restartableInit2, restartableInit1))
	})

	ginkgo.It("should terminate sidecars simultaneously if prestop doesn't exit", func() {
		restartableInit1 := "restartable-init-1"
		restartableInit2 := "restartable-init-2"
		restartableInit3 := "restartable-init-3"
		regular1 := "regular-1"

		makePrestop := func(containerName string) *v1.Lifecycle {
			return &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: ExecCommand(prefixedName(PreStopPrefix, containerName), execCommand{
							ExitCode:      0,
							ContainerName: containerName,
							LoopForever:   true,
						}),
					},
				},
			}
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "serialize-termination",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:          restartableInit1,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit1, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
						Lifecycle: makePrestop(restartableInit1),
					},
					{
						Name:          restartableInit2,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit2, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
						Lifecycle: makePrestop(restartableInit2),
					},
					{
						Name:          restartableInit3,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit3, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
						Lifecycle: makePrestop(restartableInit3),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    5,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(pod)

		client := e2epod.NewPodClient(f)
		pod = client.Create(context.TODO(), pod)

		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, 5*time.Minute)
		framework.ExpectNoError(err)

		pod, err = client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		results := parseOutput(context.TODO(), f, pod)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.StartsBefore(restartableInit1, restartableInit2))
		framework.ExpectNoError(results.StartsBefore(restartableInit1, restartableInit3))
		framework.ExpectNoError(results.StartsBefore(restartableInit2, restartableInit3))
		framework.ExpectNoError(results.StartsBefore(restartableInit1, regular1))
		framework.ExpectNoError(results.StartsBefore(restartableInit2, regular1))
		framework.ExpectNoError(results.StartsBefore(restartableInit3, regular1))

		ps1, err := results.TimeOfStart(prefixedName(PreStopPrefix, restartableInit1))
		framework.ExpectNoError(err)
		ps2, err := results.TimeOfStart(prefixedName(PreStopPrefix, restartableInit2))
		framework.ExpectNoError(err)
		ps3, err := results.TimeOfStart(prefixedName(PreStopPrefix, restartableInit3))
		framework.ExpectNoError(err)

		ps1Last, err := results.TimeOfLastLoop(prefixedName(PreStopPrefix, restartableInit1))
		framework.ExpectNoError(err)
		ps2Last, err := results.TimeOfLastLoop(prefixedName(PreStopPrefix, restartableInit2))
		framework.ExpectNoError(err)
		ps3Last, err := results.TimeOfLastLoop(prefixedName(PreStopPrefix, restartableInit3))
		framework.ExpectNoError(err)

		const simulToleration = 500 // milliseconds
		// should all end together since they loop infinitely and exceed their grace period
		gomega.Expect(ps1Last-ps2Last).To(gomega.BeNumerically("~", 0, simulToleration),
			fmt.Sprintf("expected PostStart 1 & PostStart 2 to be killed at the same time, got %s", results))
		gomega.Expect(ps1Last-ps3Last).To(gomega.BeNumerically("~", 0, simulToleration),
			fmt.Sprintf("expected PostStart 1 & PostStart 3 to be killed at the same time, got %s", results))
		gomega.Expect(ps2Last-ps3Last).To(gomega.BeNumerically("~", 0, simulToleration),
			fmt.Sprintf("expected PostStart 2 & PostStart 3 to be killed at the same time, got %s", results))

		// 30 seconds + 2 second minimum grace for the SIGKILL
		const lifetimeToleration = 1000 // milliseconds
		gomega.Expect(ps1Last-ps1).To(gomega.BeNumerically("~", 32000, lifetimeToleration),
			fmt.Sprintf("expected PostStart 1 to live for ~32 seconds, got %s", results))
		gomega.Expect(ps2Last-ps2).To(gomega.BeNumerically("~", 32000, lifetimeToleration),
			fmt.Sprintf("expected PostStart 2 to live for ~32 seconds, got %s", results))
		gomega.Expect(ps3Last-ps3).To(gomega.BeNumerically("~", 32000, lifetimeToleration),
			fmt.Sprintf("expected PostStart 3 to live for ~32 seconds, got %s", results))

	})

	ginkgo.It("should call sidecar container PreStop hook simultaneously", func() {
		restartableInit1 := "restartable-init-1"
		restartableInit2 := "restartable-init-2"
		restartableInit3 := "restartable-init-3"
		regular1 := "regular-1"

		makePrestop := func(containerName string) *v1.Lifecycle {
			return &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: ExecCommand(prefixedName(PreStopPrefix, containerName), execCommand{
							Delay:         1,
							ExitCode:      0,
							ContainerName: containerName,
						}),
					},
				},
			}
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "serialize-termination-simul-prestop",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:          restartableInit1,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit1, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
						Lifecycle: makePrestop(restartableInit1),
					},
					{
						Name:          restartableInit2,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit2, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
						Lifecycle: makePrestop(restartableInit2),
					},
					{
						Name:          restartableInit3,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit3, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
						Lifecycle: makePrestop(restartableInit3),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    5,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(pod)

		client := e2epod.NewPodClient(f)
		pod = client.Create(context.TODO(), pod)

		err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, 5*time.Minute)
		framework.ExpectNoError(err)

		pod, err = client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		results := parseOutput(context.TODO(), f, pod)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.StartsBefore(restartableInit1, restartableInit2))
		framework.ExpectNoError(results.StartsBefore(restartableInit1, restartableInit3))
		framework.ExpectNoError(results.StartsBefore(restartableInit2, restartableInit3))
		framework.ExpectNoError(results.StartsBefore(restartableInit1, regular1))
		framework.ExpectNoError(results.StartsBefore(restartableInit2, regular1))
		framework.ExpectNoError(results.StartsBefore(restartableInit3, regular1))

		// main containers exit first
		framework.ExpectNoError(results.ExitsBefore(regular1, restartableInit1))
		framework.ExpectNoError(results.ExitsBefore(regular1, restartableInit2))
		framework.ExpectNoError(results.ExitsBefore(regular1, restartableInit3))

		// followed by sidecars in reverse order
		framework.ExpectNoError(results.ExitsBefore(restartableInit3, restartableInit2))
		framework.ExpectNoError(results.ExitsBefore(restartableInit2, restartableInit1))

		// and the pre-stop hooks should have been called simultaneously
		ps1, err := results.TimeOfStart(prefixedName(PreStopPrefix, restartableInit1))
		framework.ExpectNoError(err)
		ps2, err := results.TimeOfStart(prefixedName(PreStopPrefix, restartableInit2))
		framework.ExpectNoError(err)
		ps3, err := results.TimeOfStart(prefixedName(PreStopPrefix, restartableInit3))
		framework.ExpectNoError(err)

		const toleration = 500 // milliseconds
		gomega.Expect(ps1-ps2).To(gomega.BeNumerically("~", 0, toleration),
			fmt.Sprintf("expected PostStart 1 & PostStart 2 to start at the same time, got %s", results))
		gomega.Expect(ps1-ps3).To(gomega.BeNumerically("~", 0, toleration),
			fmt.Sprintf("expected PostStart 1 & PostStart 3 to start at the same time, got %s", results))
		gomega.Expect(ps2-ps3).To(gomega.BeNumerically("~", 0, toleration),
			fmt.Sprintf("expected PostStart 2 & PostStart 3 to start at the same time, got %s", results))
	})

	ginkgo.It("should not hang in termination if terminated during initialization", func() {
		startInit := "start-init"
		restartableInit1 := "restartable-init-1"
		restartableInit2 := "restartable-init-2"
		restartableInit3 := "restartable-init-3"
		regular1 := "regular-1"

		makePrestop := func(containerName string) *v1.Lifecycle {
			return &v1.Lifecycle{
				PreStop: &v1.LifecycleHandler{
					Exec: &v1.ExecAction{
						Command: ExecCommand(prefixedName(PreStopPrefix, containerName), execCommand{
							Delay:         1,
							ExitCode:      0,
							ContainerName: containerName,
						}),
					},
				},
			}
		}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "dont-hang-if-terminated-in-init",
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyNever,
				InitContainers: []v1.Container{
					{
						Name:  startInit,
						Image: busyboxImage,
						Command: ExecCommand(startInit, execCommand{
							Delay:              300,
							TerminationSeconds: 0,
							ExitCode:           0,
						}),
					},
					{
						Name:          restartableInit1,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit1, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
						Lifecycle: makePrestop(restartableInit1),
					},
					{
						Name:          restartableInit2,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit2, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
						Lifecycle: makePrestop(restartableInit2),
					},
					{
						Name:          restartableInit3,
						Image:         busyboxImage,
						RestartPolicy: &containerRestartPolicyAlways,
						Command: ExecCommand(restartableInit3, execCommand{
							Delay:              60,
							TerminationSeconds: 5,
							ExitCode:           0,
						}),
						Lifecycle: makePrestop(restartableInit3),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    5,
							ExitCode: 0,
						}),
					},
				},
			},
		}

		preparePod(pod)

		client := e2epod.NewPodClient(f)
		pod = client.Create(context.TODO(), pod)

		err := e2epod.WaitForPodCondition(context.TODO(), f.ClientSet, pod.Namespace, pod.Name, "pod pending and init running", 2*time.Minute, func(pod *v1.Pod) (bool, error) {
			if pod.Status.Phase != v1.PodPending {
				return false, fmt.Errorf("pod should be in pending phase")
			}
			if len(pod.Status.InitContainerStatuses) < 1 {
				return false, nil
			}
			containerStatus := pod.Status.InitContainerStatuses[0]
			return *containerStatus.Started && containerStatus.State.Running != nil, nil
		})
		framework.ExpectNoError(err)

		// the init container is running, so we stop the pod before the sidecars even start
		start := time.Now()
		grace := int64(3)
		ginkgo.By("deleting the pod")
		err = client.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &grace})
		framework.ExpectNoError(err)
		ginkgo.By("waiting for the pod to disappear")
		err = e2epod.WaitForPodNotFoundInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, 120*time.Second)
		framework.ExpectNoError(err)

		buffer := int64(2)
		deleteTime := time.Since(start).Seconds()
		// should delete quickly and not try to start/wait on any sidecars since they never started
		gomega.Expect(deleteTime).To(gomega.BeNumerically("<", grace+buffer), fmt.Sprintf("should delete in < %d seconds, took %f", grace+buffer, deleteTime))
	})
})

var _ = SIGDescribe(nodefeature.SidecarContainers, framework.WithSerial(), "Containers Lifecycle", func() {
	f := framework.NewDefaultFramework("containers-lifecycle-test-serial")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should restart the containers in right order after the node reboot", func(ctx context.Context) {
		init1 := "init-1"
		restartableInit2 := "restartable-init-2"
		init3 := "init-3"
		regular1 := "regular-1"

		podLabels := map[string]string{
			"test":      "containers-lifecycle-test-serial",
			"namespace": f.Namespace.Name,
		}
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "initialized-pod",
				Labels: podLabels,
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				InitContainers: []v1.Container{
					{
						Name:  init1,
						Image: busyboxImage,
						Command: ExecCommand(init1, execCommand{
							Delay:    5,
							ExitCode: 0,
						}),
					},
					{
						Name:  restartableInit2,
						Image: busyboxImage,
						Command: ExecCommand(restartableInit2, execCommand{
							Delay:    300,
							ExitCode: 0,
						}),
						RestartPolicy: &containerRestartPolicyAlways,
					},
					{
						Name:  init3,
						Image: busyboxImage,
						Command: ExecCommand(init3, execCommand{
							Delay:    5,
							ExitCode: 0,
						}),
					},
				},
				Containers: []v1.Container{
					{
						Name:  regular1,
						Image: busyboxImage,
						Command: ExecCommand(regular1, execCommand{
							Delay:    300,
							ExitCode: 0,
						}),
					},
				},
			},
		}
		preparePod(pod)

		client := e2epod.NewPodClient(f)
		pod = client.Create(ctx, pod)
		ginkgo.By("Waiting for the pod to be initialized and run")
		err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
		framework.ExpectNoError(err)

		ginkgo.By("Getting the current pod sandbox ID")
		rs, _, err := getCRIClient()
		framework.ExpectNoError(err)

		sandboxes, err := rs.ListPodSandbox(ctx, &runtimeapi.PodSandboxFilter{
			LabelSelector: podLabels,
		})
		framework.ExpectNoError(err)
		gomega.Expect(sandboxes).To(gomega.HaveLen(1))
		podSandboxID := sandboxes[0].Id

		ginkgo.By("Stopping the kubelet")
		restartKubelet := stopKubelet()
		gomega.Eventually(ctx, func() bool {
			return kubeletHealthCheck(kubeletHealthCheckURL)
		}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeFalse())

		ginkgo.By("Stopping the pod sandbox to simulate the node reboot")
		err = rs.StopPodSandbox(ctx, podSandboxID)
		framework.ExpectNoError(err)

		ginkgo.By("Restarting the kubelet")
		restartKubelet()
		gomega.Eventually(ctx, func() bool {
			return kubeletHealthCheck(kubeletHealthCheckURL)
		}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrue())

		ginkgo.By("Waiting for the pod to be re-initialized and run")
		err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "re-initialized", f.Timeouts.PodStart, func(pod *v1.Pod) (bool, error) {
			if pod.Status.ContainerStatuses[0].RestartCount < 1 {
				return false, nil
			}
			if pod.Status.Phase != v1.PodRunning {
				return false, nil
			}
			return true, nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("Parsing results")
		pod, err = client.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		results := parseOutput(context.TODO(), f, pod)

		ginkgo.By("Analyzing results")
		init1Started, err := results.FindIndex(init1, "Started", 0)
		framework.ExpectNoError(err)
		restartableInit2Started, err := results.FindIndex(restartableInit2, "Started", 0)
		framework.ExpectNoError(err)
		init3Started, err := results.FindIndex(init3, "Started", 0)
		framework.ExpectNoError(err)
		regular1Started, err := results.FindIndex(regular1, "Started", 0)
		framework.ExpectNoError(err)

		init1Restarted, err := results.FindIndex(init1, "Started", init1Started+1)
		framework.ExpectNoError(err)
		restartableInit2Restarted, err := results.FindIndex(restartableInit2, "Started", restartableInit2Started+1)
		framework.ExpectNoError(err)
		init3Restarted, err := results.FindIndex(init3, "Started", init3Started+1)
		framework.ExpectNoError(err)
		regular1Restarted, err := results.FindIndex(regular1, "Started", regular1Started+1)
		framework.ExpectNoError(err)

		framework.ExpectNoError(init1Started.IsBefore(restartableInit2Started))
		framework.ExpectNoError(restartableInit2Started.IsBefore(init3Started))
		framework.ExpectNoError(init3Started.IsBefore(regular1Started))

		framework.ExpectNoError(init1Restarted.IsBefore(restartableInit2Restarted))
		framework.ExpectNoError(restartableInit2Restarted.IsBefore(init3Restarted))
		framework.ExpectNoError(init3Restarted.IsBefore(regular1Restarted))
	})
})
