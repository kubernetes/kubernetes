//go:build linux

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

package e2enode

import (
	"context"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"

	"github.com/onsi/ginkgo/v2"
)

// kubeletProcessName is the process name used to locate the kubelet's cgroup
// (paired with getPidsForProcess in util.go). Used by container_manager_test.go.
const kubeletProcessName = "kubelet"

// deletePodsSync deletes a list of pods and block until pods disappear.
func deletePodsSync(ctx context.Context, f *framework.Framework, pods []*v1.Pod) {
	var wg sync.WaitGroup
	for i := range pods {
		pod := pods[i]
		wg.Add(1)
		go func() {
			defer ginkgo.GinkgoRecover()
			defer wg.Done()

			e2epod.NewPodClient(f).DeleteSync(ctx, pod.ObjectMeta.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
		}()
	}
	wg.Wait()
}

// createBatchPodWithRateControl creates a batch of pods at a controlled rate,
// returning the create time per pod name.
func createBatchPodWithRateControl(ctx context.Context, f *framework.Framework, pods []*v1.Pod, interval time.Duration) map[string]metav1.Time {
	createTimes := make(map[string]metav1.Time)
	for i := range pods {
		pod := pods[i]
		createTimes[pod.ObjectMeta.Name] = metav1.Now()
		go e2epod.NewPodClient(f).Create(ctx, pod)
		time.Sleep(interval)
	}
	return createTimes
}

// newTestPods creates a list of pods (specification) for test.
func newTestPods(numPods int, volume bool, imageName, podType string) []*v1.Pod {
	var pods []*v1.Pod
	for range numPods {
		podName := "test-" + string(uuid.NewUUID())
		labels := map[string]string{
			"type": podType,
			"name": podName,
		}
		if volume {
			pods = append(pods,
				&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:   podName,
						Labels: labels,
					},
					Spec: v1.PodSpec{
						// Restart policy is always (default).
						Containers: []v1.Container{
							{
								Image: imageName,
								Name:  podName,
								VolumeMounts: []v1.VolumeMount{
									{MountPath: "/test-volume-mnt", Name: podName + "-volume"},
								},
							},
						},
						Volumes: []v1.Volume{
							{Name: podName + "-volume", VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}},
						},
					},
				})
		} else {
			pods = append(pods,
				&v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:   podName,
						Labels: labels,
					},
					Spec: v1.PodSpec{
						// Restart policy is always (default).
						Containers: []v1.Container{
							{
								Image: imageName,
								Name:  podName,
							},
						},
					},
				})
		}

	}
	return pods
}
