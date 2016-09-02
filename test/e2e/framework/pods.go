/*
Copyright 2016 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Convenience method for getting a pod client interface in the framework's namespace.
func (f *Framework) PodClient() *PodClient {
	return &PodClient{
		f:            f,
		PodInterface: f.Client.Pods(f.Namespace.Name),
	}
}

type PodClient struct {
	f *Framework
	unversioned.PodInterface
}

// Create creates a new pod according to the framework specifications (don't wait for it to start).
func (c *PodClient) Create(pod *api.Pod) *api.Pod {
	c.mungeSpec(pod)
	p, err := c.PodInterface.Create(pod)
	ExpectNoError(err, "Error creating Pod")
	return p
}

// CreateSync creates a new pod according to the framework specifications, and wait for it to start.
func (c *PodClient) CreateSync(pod *api.Pod) *api.Pod {
	p := c.Create(pod)
	ExpectNoError(c.f.WaitForPodRunning(p.Name))
	// Get the newest pod after it becomes running, some status may change after pod created, such as pod ip.
	p, err := c.Get(p.Name)
	ExpectNoError(err)
	return p
}

// CreateBatch create a batch of pods. All pods are created before waiting.
func (c *PodClient) CreateBatch(pods []*api.Pod) []*api.Pod {
	ps := make([]*api.Pod, len(pods))
	var wg sync.WaitGroup
	for i, pod := range pods {
		wg.Add(1)
		go func(i int, pod *api.Pod) {
			defer wg.Done()
			defer GinkgoRecover()
			ps[i] = c.CreateSync(pod)
		}(i, pod)
	}
	wg.Wait()
	return ps
}

// Update updates the pod object. It retries if there is a conflict, throw out error if
// there is any other errors. name is the pod name, updateFn is the function updating the
// pod object.
func (c *PodClient) Update(name string, updateFn func(pod *api.Pod)) {
	ExpectNoError(wait.Poll(time.Millisecond*500, time.Second*30, func() (bool, error) {
		pod, err := c.PodInterface.Get(name)
		if err != nil {
			return false, fmt.Errorf("failed to get pod %q: %v", name, err)
		}
		updateFn(pod)
		_, err = c.PodInterface.Update(pod)
		if err == nil {
			Logf("Successfully updated pod %q", name)
			return true, nil
		}
		if errors.IsConflict(err) {
			Logf("Conflicting update to pod %q, re-get and re-update: %v", name, err)
			return false, nil
		}
		return false, fmt.Errorf("failed to update pod %q: %v", name, err)
	}))
}

// mungeSpec apply test-suite specific transformations to the pod spec.
func (c *PodClient) mungeSpec(pod *api.Pod) {
	if TestContext.NodeName != "" {
		Expect(pod.Spec.NodeName).To(Or(BeZero(), Equal(TestContext.NodeName)), "Test misconfigured")
		pod.Spec.NodeName = TestContext.NodeName
	}
}

// TODO(random-liu): Move pod wait function into this file

func CreateMemhogPod(f *Framework, genName string, ctnName string, res api.ResourceRequirements) *api.Pod {
	env := []api.EnvVar{
		{
			Name: "MEMORY_LIMIT",
			ValueFrom: &api.EnvVarSource{
				ResourceFieldRef: &api.ResourceFieldSelector{
					Resource: "limits.memory",
				},
			},
		},
	}

	// If there is a limit specified, pass 80% of it for -mem-total, otherwise use the downward API
	// to pass limits.memory, which will be the total memory available.
	// This helps prevent a guaranteed pod from triggering an OOM kill due to it's low memory limit,
	// which will cause the test to fail inappropriately.
	var memLimit string
	if limit, ok := res.Limits["memory"]; ok {
		memLimit = strconv.Itoa(int(
			float64(limit.Value()) * 0.8))
	} else {
		memLimit = "$(MEMORY_LIMIT)"
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			GenerateName: genName,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyNever,
			Containers: []api.Container{
				{
					Name:            ctnName,
					Image:           "gcr.io/google-containers/stress:v1",
					ImagePullPolicy: "Always",
					Env:             env,
					// 60 min timeout * 60s / tick per 10s = 360 ticks before timeout => ~11.11Mi/tick
					// to fill ~4Gi of memory, so initial ballpark 12Mi/tick.
					// We might see flakes due to timeout if the total memory on the nodes increases.
					Args:      []string{"-mem-alloc-size", "12Mi", "-mem-alloc-sleep", "10s", "-mem-total", memLimit},
					Resources: res,
				},
			},
		},
	}
	// The generated pod.Name will be on the pod spec returned by CreateSync
	pod = f.PodClient().CreateSync(pod)
	glog.Infof("pod created with name: %s", pod.Name)
	return pod
}
