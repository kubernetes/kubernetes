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

package e2e_node

import (
	"fmt"
	"strconv"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Eviction Policy is described here:
// https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/kubelet-eviction.md

var _ = framework.KubeDescribe("MemoryEviction [Slow] [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("eviction-test")

	Context("When there is memory pressure", func() {
		It("It should evict pods in the correct order (besteffort first, then burstable, then guaranteed)", func() {
			By("Creating a guaranteed pod, a burstable pod, and a besteffort pod.")

			// A pod is guaranteed only when requests and limits are specified for all the containers and they are equal.
			guaranteed := createMemhogPod(f, "guaranteed-", "guaranteed", api.ResourceRequirements{
				Requests: api.ResourceList{
					"cpu":    resource.MustParse("100m"),
					"memory": resource.MustParse("100Mi"),
				},
				Limits: api.ResourceList{
					"cpu":    resource.MustParse("100m"),
					"memory": resource.MustParse("100Mi"),
				}})

			// A pod is burstable if limits and requests do not match across all containers.
			burstable := createMemhogPod(f, "burstable-", "burstable", api.ResourceRequirements{
				Requests: api.ResourceList{
					"cpu":    resource.MustParse("100m"),
					"memory": resource.MustParse("100Mi"),
				}})

			// A pod is besteffort if none of its containers have specified any requests or limits.
			besteffort := createMemhogPod(f, "besteffort-", "besteffort", api.ResourceRequirements{})

			// We poll until timeout or all pods are killed.
			// Inside the func, we check that all pods are in a valid phase with
			// respect to the eviction order of best effort, then burstable, then guaranteed.
			By("Polling the Status.Phase of each pod and checking for violations of the eviction order.")
			Eventually(func() bool {

				gteed, gtErr := f.Client.Pods(f.Namespace.Name).Get(guaranteed.Name)
				framework.ExpectNoError(gtErr, fmt.Sprintf("getting pod %s", guaranteed.Name))
				gteedPh := gteed.Status.Phase

				burst, buErr := f.Client.Pods(f.Namespace.Name).Get(burstable.Name)
				framework.ExpectNoError(buErr, fmt.Sprintf("getting pod %s", burstable.Name))
				burstPh := burst.Status.Phase

				best, beErr := f.Client.Pods(f.Namespace.Name).Get(besteffort.Name)
				framework.ExpectNoError(beErr, fmt.Sprintf("getting pod %s", besteffort.Name))
				bestPh := best.Status.Phase

				glog.Infof("Pod phase: guaranteed: %v, burstable: %v, besteffort: %v", gteedPh, burstPh, bestPh)

				if bestPh == api.PodRunning {
					Expect(burstPh).NotTo(Equal(api.PodFailed), "Burstable pod failed before best effort pod")
					Expect(gteedPh).NotTo(Equal(api.PodFailed), "Guaranteed pod failed before best effort pod")
				} else if burstPh == api.PodRunning {
					Expect(gteedPh).NotTo(Equal(api.PodFailed), "Guaranteed pod failed before burstable pod")
				}

				// When both besteffort and burstable have been evicted, return true, else false
				if bestPh == api.PodFailed && burstPh == api.PodFailed {
					return true
				}
				return false

			}, 60*time.Minute, 5*time.Second).Should(Equal(true))

		})
	})

})

func createMemhogPod(f *framework.Framework, genName string, ctnName string, res api.ResourceRequirements) *api.Pod {
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
