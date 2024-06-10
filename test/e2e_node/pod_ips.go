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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

var _ = common.SIGDescribe("Pod IPs", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("pod-ips")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when pod gets terminated", func() {
		ginkgo.It("should contain podIPs in status for terminal pod", func(ctx context.Context) {
			podName := "pod-ips-" + string(uuid.NewUUID())

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

			w := &cache.ListWatch{
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return f.ClientSet.CoreV1().Pods(f.Namespace.Name).Watch(ctx, options)
				},
			}
			podsList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err, "Failed to list pods in namespace: %s", f.Namespace.Name)

			ginkgo.By(fmt.Sprintf("creating the pod (%v/%v)", podSpec.Namespace, podSpec.Name))
			podClient := e2epod.NewPodClient(f)
			pod := podClient.Create(ctx, podSpec)

			ctxUntil, cancel := context.WithTimeout(ctx, f.Timeouts.PodStart)
			defer cancel()

			ginkgo.By(fmt.Sprintf("Started watch for pod (%v/%v) to enter terminal phase", pod.Namespace, pod.Name))
			_, err = watchtools.Until(ctxUntil, podsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
				if pod, ok := event.Object.(*v1.Pod); ok {
					found := pod.ObjectMeta.Name == podName &&
						pod.ObjectMeta.Namespace == f.Namespace.Name
					if !found {
						ginkgo.By(fmt.Sprintf("Observed Pod (%s/%s) in phase %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase))
						return false, nil
					}
					ginkgo.By(fmt.Sprintf("Found Pod (%s/%s) in phase %v, podIP=%v, podIPs=%v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase, pod.Status.PodIP, pod.Status.PodIPs))
					if pod.Status.Phase != v1.PodPending {
						if len(pod.Status.PodIP) == 0 {
							framework.Failf("PodIP not set for pod (%s/%s) in phase %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase)
						}
						if len(pod.Status.PodIPs) == 0 {
							framework.Failf("PodIPs not set for pod (%s/%s) in phase %v", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name, pod.Status.Phase)
						}
					}
					// end the watch if the pod reached terminal phase
					return podutil.IsPodPhaseTerminal(pod.Status.Phase), nil
				}
				ginkgo.By(fmt.Sprintf("Observed event: %+v", event.Object))
				return false, nil
			})
			framework.ExpectNoError(err, "failed to see event that pod (%s/%s) enter terminal phase: %v", pod.Namespace, pod.Name, err)
			ginkgo.By(fmt.Sprintf("Ended watch for pod (%v/%v) entering terminal phase", pod.Namespace, pod.Name))
		})
	})
})
