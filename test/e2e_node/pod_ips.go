/*
Copyright 2024 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = common.SIGDescribe("Pod IPs", func() {
	f := framework.NewDefaultFramework("pod-ips")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelRestricted
	testFinalizer := "example.com/test-finalizer"

	watchPodIPWhileTerminating := func(ctx context.Context, pod *v1.Pod) {
		ctxUntil, cancel := context.WithTimeout(ctx, f.Timeouts.PodStart)
		defer cancel()

		fieldSelector := fields.OneTermEqualSelector("metadata.name", pod.Name).String()
		w := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.FieldSelector = fieldSelector
				return f.ClientSet.CoreV1().Pods(f.Namespace.Name).Watch(ctx, options)
			},
		}

		ginkgo.By(fmt.Sprintf("Started watch for pod (%v/%v) to enter terminal phase", pod.Namespace, pod.Name))
		_, err := watchtools.Until(ctxUntil, pod.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if pod, ok := event.Object.(*v1.Pod); ok {
				found := pod.ObjectMeta.Name == pod.Name &&
					pod.ObjectMeta.Namespace == f.Namespace.Name
				if !found {
					ginkgo.By(fmt.Sprintf("Found unexpected Pod (%s/%s) in phase %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase))
					return false, nil
				}
				ginkgo.By(fmt.Sprintf("Found Pod (%s/%s) in phase %v, podIP=%v, podIPs=%v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase, pod.Status.PodIP, pod.Status.PodIPs))
				if pod.Status.Phase != v1.PodPending {
					gomega.Expect(pod.Status.PodIP).NotTo(gomega.BeEmpty(), fmt.Sprintf("PodIP not set for pod (%s/%s) in phase %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase))
					gomega.Expect(pod.Status.PodIPs).NotTo(gomega.BeEmpty(), fmt.Sprintf("PodIPs not set for pod (%s/%s) in phase %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase))
				}
				// end the watch if the pod reached terminal phase
				return podutil.IsPodPhaseTerminal(pod.Status.Phase), nil
			}
			ginkgo.By(fmt.Sprintf("Observed event: %+v", event.Object))
			return false, nil
		})
		framework.ExpectNoError(err, "failed to see event that pod (%s/%s) enter terminal phase: %v", pod.Namespace, pod.Name, err)
		ginkgo.By(fmt.Sprintf("Ended watch for pod (%v/%v) entering terminal phase", pod.Namespace, pod.Name))
	}

	ginkgo.Context("when pod gets terminated", func() {
		ginkgo.It("should contain podIPs in status for succeeded pod", func(ctx context.Context) {
			podName := "pod-ips-success-" + string(uuid.NewUUID())

			podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:    podName,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c"},
							Args: []string{`
								sleep 1
								exit 0
							`,
							},
						},
					},
				},
			})

			ginkgo.By(fmt.Sprintf("creating the pod (%v/%v)", podSpec.Namespace, podSpec.Name))
			podClient := e2epod.NewPodClient(f)
			pod := podClient.Create(ctx, podSpec)

			watchPodIPWhileTerminating(ctx, pod)

			ginkgo.By(fmt.Sprintf("getting the terminal state for pod (%v/%v)", podSpec.Namespace, podSpec.Name))
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podName, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to get the final state for pod (%s/%s)", pod.Namespace, pod.Name)
			gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodSucceeded), fmt.Sprintf("Non-terminal phase for pod (%s/%s): %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase))
			gomega.Expect(pod.Status.ContainerStatuses[0].State.Terminated).ShouldNot(gomega.BeNil())
			gomega.Expect(pod.Status.ContainerStatuses[0].State.Terminated.ExitCode).Should(gomega.Equal(int32(0)))
		})

		ginkgo.It("should contain podIPs in status for failed pod", func(ctx context.Context) {
			podName := "pod-ips-crash-" + string(uuid.NewUUID())

			podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:    podName,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c"},
							Args: []string{`
								exit 42
							`,
							},
						},
					},
				},
			})

			ginkgo.By(fmt.Sprintf("creating the pod (%v/%v)", podSpec.Namespace, podSpec.Name))
			podClient := e2epod.NewPodClient(f)
			pod := podClient.Create(ctx, podSpec)

			watchPodIPWhileTerminating(ctx, pod)

			ginkgo.By(fmt.Sprintf("getting the terminal state for pod (%v/%v)", podSpec.Namespace, podSpec.Name))
			pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podName, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to get the final state for pod (%s/%s)", pod.Namespace, pod.Name)
			gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodFailed), fmt.Sprintf("Non-terminal phase for pod (%s/%s): %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase))
			gomega.Expect(pod.Status.ContainerStatuses[0].State.Terminated).ShouldNot(gomega.BeNil())
			gomega.Expect(pod.Status.ContainerStatuses[0].State.Terminated.ExitCode).Should(gomega.Equal(int32(42)))
		})

		ginkgo.It("should contain podIPs in status for during termination", func(ctx context.Context) {
			podName := "pod-ips-terminating-" + string(uuid.NewUUID())

			podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       podName,
					Finalizers: []string{testFinalizer},
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:    podName,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c"},
							Args: []string{`
							sleep 9999999 &
							PID=$!
							_term() {
								kill $PID
								echo "Caught SIGTERM signal!"
							}

							trap _term SIGTERM
							touch /tmp/trap-marker
							wait $PID

							exit 42
							`,
							},
							ReadinessProbe: &v1.Probe{
								PeriodSeconds: 1,
								ProbeHandler: v1.ProbeHandler{
									Exec: &v1.ExecAction{
										Command: []string{"/bin/sh", "-c", "cat /tmp/trap-marker"},
									},
								},
							},
							Lifecycle: &v1.Lifecycle{
								PreStop: &v1.LifecycleHandler{
									Exec: &v1.ExecAction{
										Command: []string{"/bin/sh", "sleep 1"},
									},
								},
							},
						},
					},
				},
			})

			ginkgo.By(fmt.Sprintf("creating the pod (%v/%v)", podSpec.Namespace, podSpec.Name))
			podClient := e2epod.NewPodClient(f)
			pod := podClient.Create(ctx, podSpec)

			ginkgo.By(fmt.Sprintf("set up cleanup of the finalizer for the pod (%v/%v)", f.Namespace.Name, pod.Name))
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).RemoveFinalizer, pod.Name, testFinalizer)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be running and the SIGTERM trap is registered", pod.Namespace, pod.Name))
			err := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, f.Timeouts.PodStart)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to trigger termination", pod.Namespace, pod.Name))
			err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

			watchPodIPWhileTerminating(ctx, pod)

			ginkgo.By(fmt.Sprintf("getting the terminal state for pod (%v/%v)", podSpec.Namespace, podSpec.Name))
			pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podName, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to get the final state for pod (%s/%s)", pod.Namespace, pod.Name)
			gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodFailed), fmt.Sprintf("Non-terminal phase for pod (%s/%s): %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase))
			gomega.Expect(pod.Status.ContainerStatuses[0].State.Terminated).ShouldNot(gomega.BeNil())
			gomega.Expect(pod.Status.ContainerStatuses[0].State.Terminated.ExitCode).Should(gomega.Equal(int32(42)))
		})
	})
})
