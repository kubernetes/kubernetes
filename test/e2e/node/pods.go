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
	"fmt"
	"math/rand"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Pods Extended", func() {
	f := framework.NewDefaultFramework("pods")

	framework.KubeDescribe("Delete Grace Period", func() {
		var podClient *framework.PodClient
		ginkgo.BeforeEach(func() {
			podClient = f.PodClient()
		})

		/*
			Release: v1.15
			Testname: Pods, delete grace period
			Description: Create a pod, make sure it is running. Using the http client send a 'delete' with gracePeriodSeconds=30. Pod SHOULD get terminated within gracePeriodSeconds and removed from API server within a window.
		*/
		ginkgo.It("should be submitted and removed", func() {
			ginkgo.By("creating the pod")
			name := "pod-submit-remove-" + string(uuid.NewUUID())
			value := strconv.Itoa(time.Now().Nanosecond())
			pod := &v1.Pod{
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
							Name:  "agnhost",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Args:  []string{"pause"},
						},
					},
				},
			}

			ginkgo.By("setting up selector")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			pods, err := podClient.List(context.TODO(), options)
			framework.ExpectNoError(err, "failed to query for pod")
			framework.ExpectEqual(len(pods.Items), 0)
			options = metav1.ListOptions{
				LabelSelector:   selector.String(),
				ResourceVersion: pods.ListMeta.ResourceVersion,
			}

			ginkgo.By("submitting the pod to kubernetes")
			podClient.Create(pod)

			ginkgo.By("verifying the pod is in kubernetes")
			selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options = metav1.ListOptions{LabelSelector: selector.String()}
			pods, err = podClient.List(context.TODO(), options)
			framework.ExpectNoError(err, "failed to query for pod")
			framework.ExpectEqual(len(pods.Items), 1)

			// We need to wait for the pod to be running, otherwise the deletion
			// may be carried out immediately rather than gracefully.
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name))
			// save the running pod
			pod, err = podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to GET scheduled pod")

			ginkgo.By("deleting the pod gracefully")
			var lastPod v1.Pod
			var statusCode int
			err = f.ClientSet.CoreV1().RESTClient().Delete().AbsPath("/api/v1/namespaces", pod.Namespace, "pods", pod.Name).Param("gracePeriodSeconds", "30").Do(context.TODO()).StatusCode(&statusCode).Into(&lastPod)
			framework.ExpectNoError(err, "failed to use http client to send delete")
			framework.ExpectEqual(statusCode, http.StatusOK, "failed to delete gracefully by client request")

			ginkgo.By("verifying the kubelet observed the termination notice")

			// allow up to 3x grace period (which allows process termination)
			// for the kubelet to remove from api.  need to follow-up on if this
			// latency between termination and reportal can be isolated further.
			start := time.Now()
			err = wait.Poll(time.Second*5, time.Second*30*3, func() (bool, error) {
				podList, err := e2ekubelet.GetKubeletPods(f.ClientSet, pod.Spec.NodeName)
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

			framework.ExpectNotEqual(lastPod.DeletionTimestamp, nil)
			framework.ExpectNotEqual(lastPod.Spec.TerminationGracePeriodSeconds, 0)

			selector = labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))
			options = metav1.ListOptions{LabelSelector: selector.String()}
			pods, err = podClient.List(context.TODO(), options)
			framework.ExpectNoError(err, "failed to query for pods")
			framework.ExpectEqual(len(pods.Items), 0)

		})
	})

	framework.KubeDescribe("Pods Set QOS Class", func() {
		var podClient *framework.PodClient
		ginkgo.BeforeEach(func() {
			podClient = f.PodClient()
		})

		/*
			Release: v1.9
			Testname: Pods, QOS
			Description:  Create a Pod with CPU and Memory request and limits. Pod status MUST have QOSClass set to PodQOSGuaranteed.
			Behaviors:
			- pod/spec/container/resources
		*/
		framework.ConformanceIt("should be set on Pods with matching resource requests and limits for memory and cpu", func() {
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
			podClient.Create(pod)

			ginkgo.By("verifying QOS class is set on the pod")
			pod, err := podClient.Get(context.TODO(), name, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to query for pod")
			framework.ExpectEqual(pod.Status.QOSClass, v1.PodQOSGuaranteed)
		})
	})

	framework.KubeDescribe("Pod Container Status", func() {
		var podClient *framework.PodClient
		ginkgo.BeforeEach(func() {
			podClient = f.PodClient()
		})

		ginkgo.It("should never report success for a pending container", func() {
			ginkgo.By("creating pods that should always exit 1 and terminating the pod after a random delay")

			var reBug88766 = regexp.MustCompile(`rootfs_linux.*kubernetes\.io~secret.*no such file or directory`)

			var (
				lock sync.Mutex
				errs []error

				wg sync.WaitGroup
			)

			const delay = 2000
			const workers = 3
			const pods = 15
			var min, max time.Duration
			for i := 0; i < workers; i++ {
				wg.Add(1)
				go func(i int) {
					defer ginkgo.GinkgoRecover()
					defer wg.Done()
					for retries := 0; retries < pods; retries++ {
						name := fmt.Sprintf("pod-submit-status-%d-%d", i, retries)
						value := strconv.Itoa(time.Now().Nanosecond())
						one := int64(1)
						pod := &v1.Pod{
							ObjectMeta: metav1.ObjectMeta{
								Name: name,
								Labels: map[string]string{
									"name": "foo",
									"time": value,
								},
							},
							Spec: v1.PodSpec{
								RestartPolicy:                 v1.RestartPolicyNever,
								TerminationGracePeriodSeconds: &one,
								Containers: []v1.Container{
									{
										Name:  "busybox",
										Image: imageutils.GetE2EImage(imageutils.BusyBox),
										Command: []string{
											"/bin/false",
										},
										Resources: v1.ResourceRequirements{
											Requests: v1.ResourceList{
												v1.ResourceCPU:    resource.MustParse("5m"),
												v1.ResourceMemory: resource.MustParse("10Mi"),
											},
										},
									},
								},
							},
						}

						// create the pod, capture the change events, then delete the pod
						start := time.Now()
						created := podClient.Create(pod)
						ch := make(chan []watch.Event)
						go func() {
							defer ginkgo.GinkgoRecover()
							defer close(ch)
							w, err := podClient.Watch(context.TODO(), metav1.ListOptions{
								ResourceVersion: created.ResourceVersion,
								FieldSelector:   fmt.Sprintf("metadata.name=%s", pod.Name),
							})
							if err != nil {
								framework.Logf("Unable to watch pod %s: %v", pod.Name, err)
								return
							}
							defer w.Stop()
							events := []watch.Event{
								{Type: watch.Added, Object: created},
							}
							for event := range w.ResultChan() {
								events = append(events, event)
								if event.Type == watch.Error {
									framework.Logf("watch error seen for %s: %#v", pod.Name, event.Object)
								}
								if event.Type == watch.Deleted {
									framework.Logf("watch delete seen for %s", pod.Name)
									break
								}
							}
							ch <- events
						}()

						t := time.Duration(rand.Intn(delay)) * time.Millisecond
						time.Sleep(t)
						err := podClient.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
						framework.ExpectNoError(err, "failed to delete pod")

						var (
							events []watch.Event
							ok     bool
						)
						select {
						case events, ok = <-ch:
							if !ok {
								continue
							}
							if len(events) < 2 {
								framework.Fail("only got a single event")
							}
						case <-time.After(5 * time.Minute):
							framework.Failf("timed out waiting for watch events for %s", pod.Name)
						}

						end := time.Now()

						// check the returned events for consistency
						var duration, completeDuration time.Duration
						var hasContainers, hasTerminated, hasTerminalPhase, hasRunningContainers bool
						verifyFn := func(event watch.Event) error {
							var ok bool
							pod, ok = event.Object.(*v1.Pod)
							if !ok {
								framework.Logf("Unexpected event object: %s %#v", event.Type, event.Object)
								return nil
							}

							if len(pod.Status.InitContainerStatuses) != 0 {
								return fmt.Errorf("pod %s on node %s had incorrect init containers: %#v", pod.Name, pod.Spec.NodeName, pod.Status.InitContainerStatuses)
							}
							if len(pod.Status.ContainerStatuses) == 0 {
								if hasContainers {
									return fmt.Errorf("pod %s on node %s had incorrect containers: %#v", pod.Name, pod.Spec.NodeName, pod.Status.ContainerStatuses)
								}
								return nil
							}
							hasContainers = true
							if len(pod.Status.ContainerStatuses) != 1 {
								return fmt.Errorf("pod %s on node %s had incorrect containers: %#v", pod.Name, pod.Spec.NodeName, pod.Status.ContainerStatuses)
							}
							status := pod.Status.ContainerStatuses[0]
							t := status.State.Terminated
							if hasTerminated {
								if status.State.Waiting != nil || status.State.Running != nil {
									return fmt.Errorf("pod %s on node %s was terminated and then changed state: %#v", pod.Name, pod.Spec.NodeName, status)
								}
								if t == nil {
									return fmt.Errorf("pod %s on node %s was terminated and then had termination cleared: %#v", pod.Name, pod.Spec.NodeName, status)
								}
							}
							hasRunningContainers = status.State.Waiting == nil && status.State.Terminated == nil
							if t != nil {
								if !t.FinishedAt.Time.IsZero() {
									duration = t.FinishedAt.Sub(t.StartedAt.Time)
									completeDuration = t.FinishedAt.Sub(pod.CreationTimestamp.Time)
								}

								defer func() { hasTerminated = true }()
								switch {
								case t.ExitCode == 1:
									// expected
								case t.ExitCode == 128 && (t.Reason == "StartError" || t.Reason == "ContainerCannotRun") && reBug88766.MatchString(t.Message):
									// pod volume teardown races with container start in CRI, which reports a failure
									framework.Logf("pod %s on node %s failed with the symptoms of https://github.com/kubernetes/kubernetes/issues/88766", pod.Name, pod.Spec.NodeName)
								default:
									return fmt.Errorf("pod %s on node %s container unexpected exit code %d: start=%s end=%s reason=%s message=%s", pod.Name, pod.Spec.NodeName, t.ExitCode, t.StartedAt, t.FinishedAt, t.Reason, t.Message)
								}
							}
							if pod.Status.Phase == v1.PodFailed || pod.Status.Phase == v1.PodSucceeded {
								hasTerminalPhase = true
							} else {
								if hasTerminalPhase {
									return fmt.Errorf("pod %s on node %s was in a terminal phase and then reverted: %#v", pod.Name, pod.Spec.NodeName, pod.Status)
								}
							}
							return nil
						}

						var eventErr error
						for _, event := range events[1:] {
							if err := verifyFn(event); err != nil {
								eventErr = err
								break
							}
						}
						func() {
							lock.Lock()
							defer lock.Unlock()

							if eventErr != nil {
								errs = append(errs, eventErr)
								return
							}

							if !hasTerminalPhase {
								var names []string
								for _, status := range pod.Status.ContainerStatuses {
									if status.State.Terminated != nil || status.State.Running != nil {
										names = append(names, status.Name)
									}
								}
								switch {
								case len(names) > 0:
									errs = append(errs, fmt.Errorf("pod %s on node %s did not reach a terminal phase before being deleted but had running containers: phase=%s, running-containers=%s", pod.Name, pod.Spec.NodeName, pod.Status.Phase, strings.Join(names, ",")))
								case pod.Status.Phase != v1.PodPending:
									errs = append(errs, fmt.Errorf("pod %s on node %s was not Pending but has no running containers: phase=%s", pod.Name, pod.Spec.NodeName, pod.Status.Phase))
								}
							}
							if hasRunningContainers {
								data, _ := json.MarshalIndent(pod.Status.ContainerStatuses, "", "  ")
								errs = append(errs, fmt.Errorf("pod %s on node %s had running or unknown container status before being deleted:\n%s", pod.Name, pod.Spec.NodeName, string(data)))
							}
						}()

						if duration < min {
							min = duration
						}
						if duration > max || max == 0 {
							max = duration
						}
						framework.Logf("Pod %s on node %s timings total=%s t=%s run=%s execute=%s", pod.Name, pod.Spec.NodeName, end.Sub(start), t, completeDuration, duration)
					}

				}(i)
			}

			wg.Wait()

			if len(errs) > 0 {
				var messages []string
				for _, err := range errs {
					messages = append(messages, err.Error())
				}
				framework.Failf("%d errors:\n%v", len(errs), strings.Join(messages, "\n"))
			}
		})

	})

	framework.KubeDescribe("Pod Container lifecycle", func() {
		var podClient *framework.PodClient
		ginkgo.BeforeEach(func() {
			podClient = f.PodClient()
		})

		ginkgo.It("should not create extra sandbox if all containers are done", func() {
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
			createdPod := podClient.Create(pod)
			defer func() {
				ginkgo.By("deleting the pod")
				podClient.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})
			}()

			framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, f.Namespace.Name))

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
				eventList, err = f.ClientSet.CoreV1().Events(f.Namespace.Name).List(context.TODO(), options)
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
	})
})
