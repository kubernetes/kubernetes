/*
Copyright 2021 The Kubernetes Authors.

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
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("ContainerRestartBackoff", func() {
	f := framework.NewDefaultFramework("container-restart-backoff-test")
	ginkgo.Context("when a container restart", func() {

		/*
		  Release: v1.23
		  Testname: Kubelet, Container, Restart, Backoff
		  Description: Restarted the container with an exponential backoff.
		*/
		framework.ConformanceIt("Should be restarted the container with an exponential backoff [NodeConformance]", func() {
			ginkgo.By("create container")
			name := "test-container-restart-backoff"
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyAlways,
					Containers: []v1.Container{
						{
							Name:  "c",
							Image: framework.BusyBoxImage,
							Command: []string{
								"false",
							},
						},
					},
				},
			}
			f.PodClient().Create(pod)
			defer f.PodClient().Delete(context.Background(), pod.Name, metav1.DeleteOptions{})

			watcher, err := f.PodClient().Watch(context.Background(), metav1.ListOptions{
				FieldSelector: fields.OneTermEqualSelector("metadata.name", pod.Name).String(),
			})
			framework.ExpectNoError(err, "watch pod")
			defer watcher.Stop()

			backOffPeriod := 10 // pkg/kubelet.backOffPeriod
			expectRestartBackoffSecond := []int{
				0,
				10, // Period duration of retreat backoff from pkg/kubelet.backOffPeriod
				10,
				10, // Initial duration of retreat backoff from pkg/kubelet.backOffPeriod
				20,
				40,
				80,
				160,
				300, // Maximum duration of retreat backoff from pkg/kubelet.MaxContainerBackOff
				300,
				300,
				300,
			}

			ginkgo.By("watch container restart")
			var count int32
			var lastAt time.Time
			for result := range watcher.ResultChan() {
				pod := result.Object.(*v1.Pod)
				if pod.Name != name {
					continue
				}
				if len(pod.Status.ContainerStatuses) == 0 {
					continue
				}
				status := pod.Status.ContainerStatuses[0]
				terminated := status.LastTerminationState.Terminated
				if terminated == nil {
					continue
				}
				if status.RestartCount <= count {
					continue
				}
				if lastAt.IsZero() && pod.Status.StartTime != nil {
					lastAt = pod.Status.StartTime.Time
				}

				count = status.RestartCount
				if int(count) >= len(expectRestartBackoffSecond) {
					break
				}

				// Get actual tolerance
				gotIntervalSecond := int(terminated.StartedAt.Sub(lastAt) / time.Second)
				wantIntervalSecond := expectRestartBackoffSecond[count]
				gotTolerance := gotIntervalSecond - wantIntervalSecond
				if gotTolerance < 0 {
					gotTolerance = -gotTolerance
				}

				framework.Logf("container restart backoff: count:%d interval:%ds startdAt:%s", count, gotIntervalSecond, terminated.StartedAt)

				// Theoretically, the maximum tolerance is backOffPeriod, but in order to avoid the flakes from increasing the tolerance
				tolerance := backOffPeriod * 2
				if tolerance < gotTolerance {
					framework.Failf("container restart backoff does not meet expectations tolerance, want:%ds got:%ds", tolerance, gotTolerance)
				}

				lastAt = terminated.StartedAt.Time
			}
		})
	})
})
