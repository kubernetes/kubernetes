/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
)

// This test requires that --terminated-pod-gc-threshold=100 be set on the controller manager
var _ = Describe("Garbage collector", func() {
	f := NewFramework("garbage-collector")
	It("should handle the creation of 1000 pods", func() {
		SkipUnlessProviderIs("gce")

		var count int
		for count < 1000 {
			pod, err := createTerminatingPod(f)
			pod.ResourceVersion = ""
			pod.Status.Phase = api.PodFailed
			pod, err = f.Client.Pods(f.Namespace.Name).UpdateStatus(pod)
			if err != nil {
				Failf("err failing pod: %v", err)
			}

			count++
			if count%50 == 0 {
				Logf("count: %v", count)
			}
		}

		Logf("created: %v", count)
		// This sleep has to be longer than the gcCheckPeriod defined
		// in pkg/controller/gc/gc_controller.go which is currently
		// 20 seconds.
		time.Sleep(30 * time.Second)

		pods, err := f.Client.Pods(f.Namespace.Name).List(labels.Everything(), fields.Everything())
		Expect(err).NotTo(HaveOccurred())
		Expect(len(pods.Items)).To(BeNumerically("==", 100))
	})
})

func createTerminatingPod(f *Framework) (*api.Pod, error) {
	uuid := util.NewUUID()
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: string(uuid),
		},
		Spec: api.PodSpec{
			NodeName: "nonexistant-node",
			Containers: []api.Container{
				{
					Name:  string(uuid),
					Image: "beta.gcr.io/google_containers/busybox",
				},
			},
		},
	}
	return f.Client.Pods(f.Namespace.Name).Create(pod)
}
