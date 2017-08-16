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
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	nodeutil "k8s.io/kubernetes/pkg/api/v1/node"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Eviction Policy is described here:
// https://github.com/kubernetes/kubernetes/blob/master/docs/proposals/kubelet-eviction.md

var _ = framework.KubeDescribe("MemoryEviction [Slow] [Serial] [Disruptive]", func() {
	const (
		evictionHard = "memory.available<40%"
	)

	f := framework.NewDefaultFramework("eviction-test")

	// This is a dummy context to wrap the outer AfterEach, which will run after the inner AfterEach.
	// We want to list all of the node and pod events, including any that occur while waiting for
	// memory pressure reduction, even if we time out while waiting.
	Context("", func() {

		AfterEach(func() {
			// Print events
			logNodeEvents(f)
			logPodEvents(f)
		})
		Context("", func() {
			tempSetCurrentKubeletConfig(f, func(c *kubeletconfig.KubeletConfiguration) {
				c.EvictionHard = evictionHard
			})

			Context("when there is memory pressure", func() {
				AfterEach(func() {
					// Wait for the memory pressure condition to disappear from the node status before continuing.
					By("waiting for the memory pressure condition on the node to disappear before ending the test.")
					Eventually(func() error {
						nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
						if err != nil {
							return fmt.Errorf("tried to get node list but got error: %v", err)
						}
						// Assuming that there is only one node, because this is a node e2e test.
						if len(nodeList.Items) != 1 {
							return fmt.Errorf("expected 1 node, but see %d. List: %v", len(nodeList.Items), nodeList.Items)
						}
						node := nodeList.Items[0]
						_, pressure := nodeutil.GetNodeCondition(&node.Status, v1.NodeMemoryPressure)
						if pressure != nil && pressure.Status == v1.ConditionTrue {
							return fmt.Errorf("node is still reporting memory pressure condition: %s", pressure)
						}
						return nil
					}, 5*time.Minute, 15*time.Second).Should(BeNil())

					// Check available memory after condition disappears, just in case:
					// Wait for available memory to decrease to a reasonable level before ending the test.
					// This helps prevent interference with tests that start immediately after this one.
					By("waiting for available memory to decrease to a reasonable level before ending the test.")
					Eventually(func() error {
						summary, err := getNodeSummary()
						if err != nil {
							return err
						}
						if summary.Node.Memory.AvailableBytes == nil {
							return fmt.Errorf("summary.Node.Memory.AvailableBytes was nil, cannot get memory stats.")
						}
						if summary.Node.Memory.WorkingSetBytes == nil {
							return fmt.Errorf("summary.Node.Memory.WorkingSetBytes was nil, cannot get memory stats.")
						}
						avail := *summary.Node.Memory.AvailableBytes
						wset := *summary.Node.Memory.WorkingSetBytes

						// memory limit = avail + wset
						limit := avail + wset
						halflimit := limit / 2

						// Wait for at least half of memory limit to be available
						if avail >= halflimit {
							return nil
						}
						return fmt.Errorf("current available memory is: %d bytes. Expected at least %d bytes available.", avail, halflimit)
					}, 5*time.Minute, 15*time.Second).Should(BeNil())

					// TODO(mtaufen): 5 minute wait to stop flaky test bleeding while we figure out what is actually going on.
					//                If related to pressure transition period in eviction manager, probably only need to wait
					//                just over 30s becasue that is the transition period set for node e2e tests. But since we
					//                know 5 min works and we don't know if transition period is the problem, wait 5 min for now.
					time.Sleep(5 * time.Minute)

					// Finally, try starting a new pod and wait for it to be scheduled and running.
					// This is the final check to try to prevent interference with subsequent tests.
					podName := "admit-best-effort-pod"
					f.PodClient().CreateSync(&v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name: podName,
						},
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyNever,
							Containers: []v1.Container{
								{
									Image: framework.GetPauseImageNameForHostArch(),
									Name:  podName,
								},
							},
						},
					})
				})

				It("should evict pods in the correct order (besteffort first, then burstable, then guaranteed)", func() {
					By("creating a guaranteed pod, a burstable pod, and a besteffort pod.")

					// A pod is guaranteed only when requests and limits are specified for all the containers and they are equal.
					guaranteed := getMemhogPod("guaranteed-pod", "guaranteed", v1.ResourceRequirements{
						Requests: v1.ResourceList{
							"cpu":    resource.MustParse("100m"),
							"memory": resource.MustParse("100Mi"),
						},
						Limits: v1.ResourceList{
							"cpu":    resource.MustParse("100m"),
							"memory": resource.MustParse("100Mi"),
						}})
					guaranteed = f.PodClient().CreateSync(guaranteed)
					glog.Infof("pod created with name: %s", guaranteed.Name)

					// A pod is burstable if limits and requests do not match across all containers.
					burstable := getMemhogPod("burstable-pod", "burstable", v1.ResourceRequirements{
						Requests: v1.ResourceList{
							"cpu":    resource.MustParse("100m"),
							"memory": resource.MustParse("100Mi"),
						}})
					burstable = f.PodClient().CreateSync(burstable)
					glog.Infof("pod created with name: %s", burstable.Name)

					// A pod is besteffort if none of its containers have specified any requests or limits	.
					besteffort := getMemhogPod("besteffort-pod", "besteffort", v1.ResourceRequirements{})
					besteffort = f.PodClient().CreateSync(besteffort)
					glog.Infof("pod created with name: %s", besteffort.Name)

					// We poll until timeout or all pods are killed.
					// Inside the func, we check that all pods are in a valid phase with
					// respect to the eviction order of best effort, then burstable, then guaranteed.
					By("polling the Status.Phase of each pod and checking for violations of the eviction order.")
					Eventually(func() error {

						gteed, gtErr := f.ClientSet.Core().Pods(f.Namespace.Name).Get(guaranteed.Name, metav1.GetOptions{})
						framework.ExpectNoError(gtErr, fmt.Sprintf("getting pod %s", guaranteed.Name))
						gteedPh := gteed.Status.Phase

						burst, buErr := f.ClientSet.Core().Pods(f.Namespace.Name).Get(burstable.Name, metav1.GetOptions{})
						framework.ExpectNoError(buErr, fmt.Sprintf("getting pod %s", burstable.Name))
						burstPh := burst.Status.Phase

						best, beErr := f.ClientSet.Core().Pods(f.Namespace.Name).Get(besteffort.Name, metav1.GetOptions{})
						framework.ExpectNoError(beErr, fmt.Sprintf("getting pod %s", besteffort.Name))
						bestPh := best.Status.Phase

						glog.Infof("pod phase: guaranteed: %v, burstable: %v, besteffort: %v", gteedPh, burstPh, bestPh)

						// NOTE/TODO(mtaufen): This should help us debug why burstable appears to fail before besteffort in some
						//                     scenarios. We have seen some evidence that the eviction manager has in fact done the
						//                     right thing and evicted the besteffort first, and attempted to change the besteffort phase
						//                     to "Failed" when it evicts it, but that for some reason the test isn't seeing the updated
						//                     phase. I'm trying to confirm or deny this.
						//                     The eviction manager starts trying to evict things when the node comes under memory
						//                     pressure, and the eviction manager reports this information in the pressure condition. If we
						//                     see the eviction manager reporting a pressure condition for a while without the besteffort failing,
						//                     and we see that the manager did in fact evict the besteffort (this should be in the Kubelet log), we
						//                     will have more reason to believe the phase is out of date.
						nodeList, err := f.ClientSet.Core().Nodes().List(metav1.ListOptions{})
						if err != nil {
							glog.Errorf("tried to get node list but got error: %v", err)
						}
						if len(nodeList.Items) != 1 {
							glog.Errorf("expected 1 node, but see %d. List: %v", len(nodeList.Items), nodeList.Items)
						}
						node := nodeList.Items[0]
						_, pressure := nodeutil.GetNodeCondition(&node.Status, v1.NodeMemoryPressure)
						glog.Infof("node pressure condition: %s", pressure)

						// NOTE/TODO(mtaufen): Also log (at least temporarily) the actual memory consumption on the node.
						//                     I used this to plot memory usage from a successful test run and it looks the
						//                     way I would expect. I want to see what the plot from a flake looks like.
						summary, err := getNodeSummary()
						if err != nil {
							return err
						}
						if summary.Node.Memory.WorkingSetBytes != nil {
							wset := *summary.Node.Memory.WorkingSetBytes
							glog.Infof("Node's working set is (bytes): %v", wset)

						}

						if bestPh == v1.PodRunning {
							Expect(burstPh).NotTo(Equal(v1.PodFailed), "burstable pod failed before best effort pod")
							Expect(gteedPh).NotTo(Equal(v1.PodFailed), "guaranteed pod failed before best effort pod")
						} else if burstPh == v1.PodRunning {
							Expect(gteedPh).NotTo(Equal(v1.PodFailed), "guaranteed pod failed before burstable pod")
						}

						// When both besteffort and burstable have been evicted, the test has completed.
						if bestPh == v1.PodFailed && burstPh == v1.PodFailed {
							return nil
						}
						return fmt.Errorf("besteffort and burstable have not yet both been evicted.")

					}, 60*time.Minute, 5*time.Second).Should(BeNil())

				})
			})
		})
	})

})

func getMemhogPod(podName string, ctnName string, res v1.ResourceRequirements) *v1.Pod {
	env := []v1.EnvVar{
		{
			Name: "MEMORY_LIMIT",
			ValueFrom: &v1.EnvVarSource{
				ResourceFieldRef: &v1.ResourceFieldSelector{
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

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
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
}
