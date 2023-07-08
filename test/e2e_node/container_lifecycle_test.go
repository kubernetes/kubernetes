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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	PostStartPrefix = "PostStart"
)

var containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways

func prefixedName(namePrefix string, name string) string {
	return fmt.Sprintf("%s-%s", namePrefix, name)
}

var _ = SIGDescribe("[NodeConformance] Containers Lifecycle ", func() {
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
		results := parseOutput(podSpec)

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
		results := parseOutput(podSpec)

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
							Delay:    2,
							ExitCode: 0,
						}),
						Lifecycle: &v1.Lifecycle{
							PostStart: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: ExecCommand(prefixedName(PostStartPrefix, regular1), execCommand{
										Delay:    1,
										ExitCode: 0,
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
		results := parseOutput(podSpec)

		ginkgo.By("Analyzing results")
		// init container should start and exit with an error, and the regular container should never start
		framework.ExpectNoError(results.StartsBefore(init1, prefixedName(PostStartPrefix, regular1)))
		framework.ExpectNoError(results.ExitsBefore(init1, prefixedName(PostStartPrefix, regular1)))

		framework.ExpectNoError(results.RunTogether(regular1, prefixedName(PostStartPrefix, regular1)))
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
		results := parseOutput(podSpec)

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
							Delay:    2,
							ExitCode: 0,
						}),
						Lifecycle: &v1.Lifecycle{
							PostStart: &v1.LifecycleHandler{
								Exec: &v1.ExecAction{
									Command: ExecCommand(prefixedName(PostStartPrefix, regular1), execCommand{
										Delay:    1,
										ExitCode: 0,
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
		results := parseOutput(podSpec)

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
				results = parseOutput(podSpec)
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
		results := parseOutput(podSpec)

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

		// To account for the time it takes to delete the pod, we add a 10 second
		// buffer. The 10 second buffer is arbitrary, but it should be enough to
		// account for the time it takes to delete the pod.
		bufferSeconds := int64(10)

		ginkgo.It("should respect termination grace period seconds [NodeConformance]", func() {
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

		ginkgo.It("should respect termination grace period seconds with long-running preStop hook [NodeConformance]", func() {
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
})

var _ = SIGDescribe("[NodeAlphaFeature:SidecarContainers] Containers Lifecycle ", func() {
	f := framework.NewDefaultFramework("containers-lifecycle-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

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

			// TODO: check for Pod to be succeeded
			err := e2epod.WaitTimeoutForPodNoLongerRunningInNamespace(context.TODO(), f.ClientSet, podSpec.Name, podSpec.Namespace, 5*time.Minute)
			framework.ExpectNoError(err)

			podSpec, err := client.Get(context.Background(), podSpec.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			results = parseOutput(podSpec)
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

		ginkgo.It("should run both restartable init cotnainers and third init container together", func() {
			framework.ExpectNoError(results.RunTogether(restartableInit2, restartableInit1))
			framework.ExpectNoError(results.RunTogether(restartableInit1, init3))
			framework.ExpectNoError(results.RunTogether(restartableInit2, init3))
		})

		ginkgo.It("should run third init container to completion before starting regular container", func() {
			framework.ExpectNoError(results.StartsBefore(init3, regular1))
			framework.ExpectNoError(results.ExitsBefore(init3, regular1))
		})

		ginkgo.It("should run both restartable init cotnainers and a regular container together", func() {
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
				results = parseOutput(podSpec)
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
				results = parseOutput(podSpec)
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
								Delay:    1,
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
				results = parseOutput(podSpec)
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
								Delay:    1,
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
				results = parseOutput(podSpec)
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
				results = parseOutput(podSpec)
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
								Delay:    1,
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
				results = parseOutput(podSpec)
			})
			ginkgo.It("should mark an Init container as failed", func() {
				framework.ExpectNoError(results.Exits(init1))
			})
			// TODO: how will we be able to test it if restartable init container
			// will never fail and there will be no termination log? Or will be?
			ginkgo.It("should be running restartable init container and a failed Init container in parallel", func() {
				framework.ExpectNoError(results.RunTogether(restartableInit1, init1))
			})
			// TODO: check preStop hooks when they are enabled
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
				results = parseOutput(podSpec)
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
				results = parseOutput(podSpec)
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
								Delay:    1,
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
				results = parseOutput(podSpec)
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
								Delay:    1,
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
				results = parseOutput(podSpec)
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
				results = parseOutput(podSpec)
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
								Delay:    1,
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
				results = parseOutput(podSpec)
			})
			ginkgo.It("should have Init container restartCount greater than 0", func() {
				framework.ExpectNoError(results.HasRestarted(init1))
			})
			// TODO: how will we be able to test it if restartable init container will never fail and there will be no termination log? Or will be?
			ginkgo.It("should be running restartable init container and a failed Init container in parallel", func() {
				framework.ExpectNoError(results.RunTogether(restartableInit1, init1))
			})
			// TODO: check preStop hooks when they are enabled
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
				results = parseOutput(podSpec)
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
				results = parseOutput(podSpec)
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
								Delay:    1,
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
				results = parseOutput(podSpec)
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
								Delay:    1,
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
				results = parseOutput(podSpec)
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
				results = parseOutput(podSpec)
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
								Delay:    1,
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
				results = parseOutput(podSpec)
			})
			ginkgo.It("should have Init container restartCount greater than 0", func() {
				framework.ExpectNoError(results.HasRestarted(init1))
			})
			// TODO: how will we be able to test it if restartable init container will never fail and there will be no termination log? Or will be?
			ginkgo.It("should be running restartable init container and a failed Init container in parallel", func() {
				framework.ExpectNoError(results.RunTogether(restartableInit1, init1))
			})
			// TODO: check preStop hooks when they are enabled
		})
	})

	ginkgo.It("should launch restartable init cotnainers serially considering the startup probe", func() {

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
		results := parseOutput(pod)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.StartsBefore(restartableInit1, restartableInit2))
		framework.ExpectNoError(results.StartsBefore(restartableInit2, regular1))
	})

	ginkgo.It("should not launch next container if the restartable init cotnainer failed to complete startup probe", func() {

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
							StartDelay: 30,
							Delay:      600,
							ExitCode:   0,
						}),
						StartupProbe: &v1.Probe{
							PeriodSeconds:    1,
							FailureThreshold: 1,
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

		ginkgo.By("Waiting for the restartable init cotnainer to restart")
		err := WaitForPodInitContainerRestartCount(context.TODO(), f.ClientSet, pod.Namespace, pod.Name, 0, 2, 2*time.Minute)
		framework.ExpectNoError(err)

		pod, err = client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		if pod.Status.Phase != v1.PodPending {
			framework.Failf("pod %q is not pending, it's %q", pod.Name, pod.Status.Phase)
		}

		results := parseOutput(pod)

		ginkgo.By("Analyzing results")
		framework.ExpectNoError(results.DoesntStart(regular1))
	})
})
