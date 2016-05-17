/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package e2e_node

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/gomega"
)

func createPod(f *framework.Framework, podName string, containers []api.Container, volumes []api.Volume) {
	podClient := f.Client.Pods(f.Namespace.Name)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
		},
		Spec: api.PodSpec{
			// Force the Pod to schedule to the node without a scheduler running
			NodeName: *nodeName,
			// Don't restart the Pod since it is expected to exit
			RestartPolicy: api.RestartPolicyNever,
			Containers:    containers,
			Volumes:       volumes,
		},
	}
	_, err := podClient.Create(pod)
	Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
	framework.ExpectNoError(f.WaitForPodRunning(pod.Name))
}

func getPauseContainer() api.Container {
	return api.Container{
		Name:  "pause",
		Image: "gcr.io/google_containers/pause:2.0",
	}
}
