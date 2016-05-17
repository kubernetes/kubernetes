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
	"time"

	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("GarbageCollect", func() {
	var cl *client.Client
	var dockerClient *docker.Client
	BeforeEach(func() {
		cl = client.NewOrDie(&client.Config{Host: *apiServerAddress})
		var err error
		dockerClient, err = docker.NewClientFromEnv()
		Expect(err).To(BeNil(), fmt.Sprintf("Error connecting to docker %v", err))
	})

	It("Should garbage collect deleted pods", func() {
		Skip("Requires docker permissions") // FIXME

		const (
			// The acceptable delta when counting containers.
			epsilon = 5
			// The number of pods to create & delete.
			numPods = 90
		)

		containers, err := dockerClient.ListContainers(docker.ListContainersOptions{All: true})
		Expect(err).To(BeNil(), fmt.Sprintf("Error listing containers %v", err))
		initialContainerCount := len(containers)

		// Start pods.
		podNames := make([]string, numPods)
		podContainers := []api.Container{getPauseContainer()}
		for i := 0; i < numPods; i++ {
			podNames[i] = fmt.Sprintf("pod-%d", i)
			createPod(cl, podNames[i], podContainers, nil)
		}

		// Wait for containers to start.
		Expect(waitForContainerCount(dockerClient, atLeast(numPods))).To(BeNil())

		// Delete pods.
		for _, podName := range podNames {
			err := cl.Pods(api.NamespaceDefault).Delete(podName, &api.DeleteOptions{})
			Expect(err).To(BeNil(), fmt.Sprintf("Error deleting Pod %q: %v", podName, err))
		}

		// Wait for containers to be garbage collected.
		Expect(waitForContainerCount(dockerClient, atMost(initialContainerCount+epsilon))).To(BeNil())
	})
})

type condition struct {
	desc string
	test func(int) bool
}

func atMost(val int) condition {
	return condition{
		desc: fmt.Sprintf("at most %d", val),
		test: func(x int) bool { return x <= val },
	}
}

func atLeast(val int) condition {
	return condition{
		desc: fmt.Sprintf("at least %d", val),
		test: func(x int) bool { return x >= val },
	}
}

// Wait for at least count containers to be running if atleast is true, or at most count containers
// to be running if atleast is false.
func waitForContainerCount(dockerClient *docker.Client, cond condition) error {
	const (
		pollPeriod  = 10 * time.Second
		pollTimeout = 5 * time.Minute
	)
	var count int
	err := wait.PollImmediate(pollPeriod, pollTimeout, func() (bool, error) {
		containers, err := dockerClient.ListContainers(docker.ListContainersOptions{All: true})
		if err != nil {
			glog.Errorf("Error listing containers: %v", err)
			return false, nil
		}
		count = len(containers)
		if cond.test(count) {
			return true, nil
		}
		glog.Infof("Waiting for %s containers, currently %d", cond.desc, count)
		return false, nil
	})
	if err != nil {
		return fmt.Errorf("timed out waiting for %s containers: found %d", cond.desc, count)
	}
	glog.Infof("Finished waiting for %s containers: currently %d", cond.desc, count)
	return nil
}
