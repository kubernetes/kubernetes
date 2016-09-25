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

package e2e

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

// This test requires that --terminated-pod-gc-threshold=100 be set on the controller manager
//
// Slow by design (7 min)
var _ = framework.KubeDescribe("Pod garbage collector [Feature:PodGarbageCollector] [Slow]", func() {
	f := framework.NewDefaultFramework("pod-garbage-collector")
	It("should handle the creation of 1000 pods", func() {
		var count int
		for count < 1000 {
			pod, err := createTerminatingPod(f)
			pod.ResourceVersion = ""
			pod.Status.Phase = api.PodFailed
			pod, err = f.Client.Pods(f.Namespace.Name).UpdateStatus(pod)
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
		var pods *api.PodList
		timeout := 2 * time.Minute
		gcThreshold := 100

		By(fmt.Sprintf("Waiting for gc controller to gc all but %d pods", gcThreshold))
		pollErr := wait.Poll(1*time.Minute, timeout, func() (bool, error) {
			pods, err = f.Client.Pods(f.Namespace.Name).List(api.ListOptions{})
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

func createTerminatingPod(f *framework.Framework) (*api.Pod, error) {
	uuid := uuid.NewUUID()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: string(uuid),
			Annotations: map[string]string{
				"scheduler.alpha.kubernetes.io/name": "please don't schedule my pods",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  string(uuid),
					Image: "gcr.io/google_containers/busybox:1.24",
				},
			},
		},
	}
	return f.Client.Pods(f.Namespace.Name).Create(pod)
}
