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

package node

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// This test requires that --terminated-pod-gc-threshold=100 be set on the controller manager
//
// Slow by design (7 min)
var _ = SIGDescribe("Pod garbage collector [Feature:PodGarbageCollector] [Slow]", func() {
	f := framework.NewDefaultFramework("pod-garbage-collector")
	ginkgo.It("should handle the creation of 1000 pods", func() {
		var count int
		for count < 1000 {
			pod, err := createTerminatingPod(f)
			if err != nil {
				framework.Failf("err creating pod: %v", err)
			}
			pod.ResourceVersion = ""
			pod.Status.Phase = v1.PodFailed
			_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).UpdateStatus(context.TODO(), pod, metav1.UpdateOptions{})
			if err != nil {
				framework.Failf("err failing pod: %v", err)
			}

			count++
			if count%50 == 0 {
				framework.Logf("count: %v", count)
			}
		}

		framework.Logf("created: %v", count)

		// The gc controller polls every 30s and fires off a goroutine per
		// pod to terminate.
		var err error
		var pods *v1.PodList
		timeout := 2 * time.Minute
		gcThreshold := 100

		ginkgo.By(fmt.Sprintf("Waiting for gc controller to gc all but %d pods", gcThreshold))
		pollErr := wait.Poll(1*time.Minute, timeout, func() (bool, error) {
			pods, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
			if err != nil {
				framework.Logf("Failed to list pod %v", err)
				return false, nil
			}
			if len(pods.Items) != gcThreshold {
				framework.Logf("Number of observed pods %v, waiting for %v", len(pods.Items), gcThreshold)
				return false, nil
			}
			return true, nil
		})
		if pollErr != nil {
			framework.Failf("Failed to GC pods within %v, %v pods remaining, error: %v", timeout, len(pods.Items), err)
		}
	})
})

func createTerminatingPod(f *framework.Framework) (*v1.Pod, error) {
	uuid := uuid.NewUUID()
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: string(uuid),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  string(uuid),
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
				},
			},
			SchedulerName: "please don't schedule my pods",
		},
	}
	return f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
}
