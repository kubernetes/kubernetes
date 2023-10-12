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
	"os"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	testFinalizer = "example.com/test-finalizer"
)

var _ = SIGDescribe("Deleted pods handling [NodeConformance]", func() {
	f := framework.NewDefaultFramework("deleted-pods-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("Should transition to Failed phase a pod which is deleted while pending", func(ctx context.Context) {
		podName := "deleted-pending-" + string(uuid.NewUUID())
		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:       podName,
				Finalizers: []string{testFinalizer},
			},
			Spec: v1.PodSpec{
				RestartPolicy: v1.RestartPolicyAlways,
				Containers: []v1.Container{
					{
						Name:            podName,
						Image:           "non-existing-repo/non-existing-image:v1.0",
						ImagePullPolicy: "Always",
						Command:         []string{"bash"},
						Args:            []string{"-c", `echo "Hello world"`},
					},
				},
			},
		}
		ginkgo.By("creating the pod with invalid image reference and finalizer")
		pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

		ginkgo.By("set up cleanup of the finalizer")
		ginkgo.DeferCleanup(e2epod.NewPodClient(f).RemoveFinalizer, pod.Name, testFinalizer)

		ginkgo.By("Waiting for the pod to be scheduled so that kubelet owns it")
		err := e2epod.WaitForPodScheduled(ctx, f.ClientSet, pod.Namespace, pod.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be scheduled: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
		err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be transitioned into the Failed phase", pod.Namespace, pod.Name))
		err = e2epod.WaitForPodTerminatedInNamespace(ctx, f.ClientSet, pod.Name, "", f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be terminated: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Fetch the end state of the pod (%v/%v)", pod.Namespace, pod.Name))
		pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to fetch the end state of the pod: %q", pod.Name)
	})

	ginkgo.DescribeTable("Should transition to Failed phase a deleted pod if non-zero exit codes",
		func(ctx context.Context, policy v1.RestartPolicy) {
			podName := "deleted-running-" + strings.ToLower(string(policy)) + "-" + string(uuid.NewUUID())
			podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       podName,
					Finalizers: []string{testFinalizer},
				},
				Spec: v1.PodSpec{
					RestartPolicy: policy,
					Containers: []v1.Container{
						{
							Name:    podName,
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sleep", "1800"},
						},
					},
				},
			})
			ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v) with restart policy: %v", f.Namespace.Name, podSpec.Name, podSpec.Spec.RestartPolicy))
			pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

			ginkgo.By("set up cleanup of the finalizer")
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).RemoveFinalizer, pod.Name, testFinalizer)

			ginkgo.By("Waiting for the pod to be running")
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
			err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, *metav1.NewDeleteOptions(1))
			framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be transitioned to the failed phase", pod.Namespace, pod.Name))
			err = e2epod.WaitForPodTerminatedInNamespace(ctx, f.ClientSet, pod.Name, "", f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be terminated: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Fetching the end state of the pod (%v/%v)", pod.Namespace, pod.Name))
			pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to fetch the end state of the pod: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Verify the pod (%v/%v) container is in the terminated state", pod.Namespace, pod.Name))
			gomega.Expect(pod.Status.ContainerStatuses).Should(gomega.HaveLen(1))
			containerStatus := pod.Status.ContainerStatuses[0]
			gomega.Expect(containerStatus.State.Terminated).ToNot(gomega.BeNil(), "The pod container is in not in the Terminated state")

			ginkgo.By(fmt.Sprintf("Verify the pod (%v/%v) container exit code is 137", pod.Namespace, pod.Name))
			gomega.Expect(containerStatus.State.Terminated.ExitCode).Should(gomega.Equal(int32(137)))
		},
		ginkgo.Entry("Restart policy Always", v1.RestartPolicyAlways),
		ginkgo.Entry("Restart policy OnFailure", v1.RestartPolicyOnFailure),
		ginkgo.Entry("Restart policy Never", v1.RestartPolicyNever),
	)

	ginkgo.DescribeTable("Should transition to Succeeded phase a deleted pod when containers complete with 0 exit code",
		func(ctx context.Context, policy v1.RestartPolicy) {
			podName := "deleted-running-" + strings.ToLower(string(policy)) + "-" + string(uuid.NewUUID())
			podSpec := e2epod.MustMixinRestrictedPodSecurity(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:       podName,
					Finalizers: []string{testFinalizer},
				},
				Spec: v1.PodSpec{
					RestartPolicy: policy,
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
							wait $PID

							exit 0
							`,
							},
						},
					},
				},
			})
			ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v) with restart policy: %v", f.Namespace.Name, podSpec.Name, podSpec.Spec.RestartPolicy))
			pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

			ginkgo.By("set up cleanup of the finalizer")
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).RemoveFinalizer, pod.Name, testFinalizer)

			ginkgo.By("Waiting for the pod to be running")
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be running: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
			// wait a little bit to make sure the we are inside the while and that the trap is registered
			time.Sleep(time.Second)
			err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be transitioned to the succeeded phase", pod.Namespace, pod.Name))
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
			framework.ExpectNoError(err, "Failed to await for the pod to be succeeded: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Fetching the end state of the pod (%v/%v)", pod.Namespace, pod.Name))
			pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Failed to fetch the end state of the pod: %q", pod.Name)

			ginkgo.By(fmt.Sprintf("Verify the pod (%v/%v) container is in the terminated state", pod.Namespace, pod.Name))
			gomega.Expect(pod.Status.ContainerStatuses).Should(gomega.HaveLen(1))
			containerStatus := pod.Status.ContainerStatuses[0]
			gomega.Expect(containerStatus.State.Terminated).ShouldNot(gomega.BeNil(), "The pod container is in not in the Terminated state")

			ginkgo.By(fmt.Sprintf("Verifying the exit code for the terminated container is 0 for pod (%v/%v)", pod.Namespace, pod.Name))
			gomega.Expect(containerStatus.State.Terminated.ExitCode).Should(gomega.Equal(int32(0)))
		},
		ginkgo.Entry("Restart policy Always", v1.RestartPolicyAlways),
		ginkgo.Entry("Restart policy OnFailure", v1.RestartPolicyOnFailure),
		ginkgo.Entry("Restart policy Never", v1.RestartPolicyNever),
	)

})

var _ = SIGDescribe("Deleted pods handling with volumes [NodeConformance]", func() {
	f := framework.NewDefaultFramework("deleted-pod-with-volume")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("Ensure volumes unmounted for deleted pods", func(ctx context.Context) {
		policy := v1.RestartPolicyAlways
		podName := "deleted-pod-with-volume-" + string(uuid.NewUUID())
		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:       podName,
				Finalizers: []string{testFinalizer},
			},
			Spec: v1.PodSpec{
				RestartPolicy: policy,
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
							echo HelloWorld >> /data/myfile

							trap _term SIGTERM
							wait $PID

							exit 0
							`,
						},
						VolumeMounts: []v1.VolumeMount{
							{
								MountPath: "/data",
								Name:      "data",
							},
						},
						SecurityContext: e2epod.GenerateContainerSecurityContext(admissionapi.LevelPrivileged),
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "data",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			},
		}

		ginkgo.By(fmt.Sprintf("Creating a pod (%v/%v) with restart policy: %v", f.Namespace.Name, podSpec.Name, podSpec.Spec.RestartPolicy))
		pod := e2epod.NewPodClient(f).Create(ctx, podSpec)

		ginkgo.By("set up cleanup of the finalizer")
		ginkgo.DeferCleanup(e2epod.NewPodClient(f).RemoveFinalizer, pod.Name, testFinalizer)

		ginkgo.By("Waiting for the pod to be running")
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be running: %q", pod.Name)

		exists := podVolumePathExists(pod.UID, "data")
		gomega.Expect(exists).Should(gomega.BeTrue(), fmt.Sprintf("Data dir should be mounted in the empty-dir for the pod (%s/%s)", pod.Namespace, pod.Name))

		podsList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err, "Failed to list pods in namespace: %s", f.Namespace.Name)

		w := &cache.ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return f.ClientSet.CoreV1().Pods(f.Namespace.Name).Watch(ctx, options)
			},
		}

		ginkgo.By(fmt.Sprintf("Deleting the pod (%v/%v) to set a deletion timestamp", pod.Namespace, pod.Name))
		// wait a little bit to make sure the we are inside the while and that the trap is registered
		time.Sleep(1 * time.Second)
		err = e2epod.NewPodClient(f).Delete(ctx, pod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Failed to delete the pod: %q", pod.Name)

		ctxUntil, cancel := context.WithTimeout(ctx, f.Timeouts.PodStart)
		defer cancel()

		ginkgo.By(fmt.Sprintf("Started watch for pod (%v/%v) to enter succeeded phase", f.Namespace.Name, pod.Name))
		_, err = watchtools.Until(ctxUntil, podsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
			if p, ok := event.Object.(*v1.Pod); ok {
				found := p.Name == pod.Name &&
					p.Namespace == f.Namespace.Name &&
					p.Status.Phase == v1.PodSucceeded
				if !found {
					ginkgo.By(fmt.Sprintf("Observed Pod (%s/%s) in phase %v", p.Namespace, p.Name, p.Status.Phase))
					return false, nil
				}
				ginkgo.By(fmt.Sprintf("Found Pod (%s/%s) in phase %v", p.Namespace, p.Name, p.Status.Phase))
				exists := podVolumePathExists(p.UID, "data")
				gomega.Expect(exists).Should(gomega.BeFalse(), fmt.Sprintf("Data dir should be unmounted for the pod (%s/%s)", p.Namespace, p.Name))
				return found, nil
			}
			ginkgo.By(fmt.Sprintf("Observed event: %+v", event.Object))
			return false, nil
		})
		framework.ExpectNoError(err, fmt.Sprintf("Failed to await for the succeeded phase for the pod (%v/%v)", pod.Namespace, pod.Name))

		ginkgo.By(fmt.Sprintf("Waiting for the pod (%v/%v) to be transitioned to the succeeded phase", pod.Namespace, pod.Name))
		err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name)
		framework.ExpectNoError(err, "Failed to await for the pod to be succeeded: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Fetching the end state of the pod (%v/%v)", pod.Namespace, pod.Name))
		pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to fetch the end state of the pod: %q", pod.Name)

		ginkgo.By(fmt.Sprintf("Verify the pod (%v/%v) container is in the terminated state", pod.Namespace, pod.Name))
		gomega.Expect(pod.Status.ContainerStatuses).Should(gomega.HaveLen(1))
		containerStatus := pod.Status.ContainerStatuses[0]
		gomega.Expect(containerStatus.State.Terminated).ShouldNot(gomega.BeNil(), "The pod container is in not in the Terminated state")

		ginkgo.By(fmt.Sprintf("Verifying the exit code for the terminated container is 0 for pod (%v/%v)", pod.Namespace, pod.Name))
		gomega.Expect(containerStatus.State.Terminated.ExitCode).Should(gomega.Equal(int32(0)))
	})
})

func podVolumePathExists(uid types.UID, dirName string) bool {
	podVolumePath := fmt.Sprintf("/var/lib/kubelet/pods/%s/volumes/kubernetes.io~empty-dir/%s", uid, dirName)
	_, err := os.Stat(podVolumePath)
	return !os.IsNotExist(err)
}
