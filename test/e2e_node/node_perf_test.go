/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e_node/perf/workloads"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

// makeNodePerfPod returns a pod with the information provided from the workload.
func makeNodePerfPod(w workloads.NodePerfWorkload) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("%s-pod", w.Name()),
		},
		Spec: w.PodSpec(),
	}
}

func setKubeletConfig(f *framework.Framework, cfg *kubeletconfig.KubeletConfiguration) {
	if cfg != nil {
		framework.ExpectNoError(setKubeletConfiguration(f, cfg))
	}

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodes, err := e2enode.TotalReady(f.ClientSet)
		framework.ExpectNoError(err)
		return nodes == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())
}

// Serial because the test updates kubelet configuration.
// Slow by design.
var _ = SIGDescribe("Node Performance Testing [Serial] [Slow]", func() {
	f := framework.NewDefaultFramework("node-performance-testing")
	var (
		wl     workloads.NodePerfWorkload
		oldCfg *kubeletconfig.KubeletConfiguration
		newCfg *kubeletconfig.KubeletConfiguration
		pod    *v1.Pod
	)
	ginkgo.JustBeforeEach(func() {
		err := wl.PreTestExec()
		framework.ExpectNoError(err)
		oldCfg, err = getCurrentKubeletConfig()
		framework.ExpectNoError(err)
		newCfg, err = wl.KubeletConfig(oldCfg)
		framework.ExpectNoError(err)
		setKubeletConfig(f, newCfg)
	})

	cleanup := func() {
		gp := int64(0)
		delOpts := metav1.DeleteOptions{
			GracePeriodSeconds: &gp,
		}
		f.PodClient().DeleteSync(pod.Name, delOpts, framework.DefaultPodDeletionTimeout)

		// We are going to give some more time for the CPU manager to do any clean
		// up it needs to do now that the pod has been deleted. Otherwise we may
		// run into a data race condition in which the PostTestExec function
		// deletes the CPU manager's checkpoint file while the CPU manager is still
		// doing work and we end with a new checkpoint file after PosttestExec has
		// finished. This issues would result in the kubelet panicking after we try
		// and set the kubelet config.
		time.Sleep(15 * time.Second)
		ginkgo.By("running the post test exec from the workload")
		err := wl.PostTestExec()
		framework.ExpectNoError(err)
		setKubeletConfig(f, oldCfg)
	}

	runWorkload := func() {
		ginkgo.By("running the workload and waiting for success")
		// Make the pod for the workload.
		pod = makeNodePerfPod(wl)
		// Create the pod.
		pod = f.PodClient().CreateSync(pod)
		// Wait for pod success.
		f.PodClient().WaitForSuccess(pod.Name, wl.Timeout())
		podLogs, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err)
		perf, err := wl.ExtractPerformanceFromLogs(podLogs)
		framework.ExpectNoError(err)
		framework.Logf("Time to complete workload %s: %v", wl.Name(), perf)
	}

	ginkgo.BeforeEach(func() {
		ginkgo.By("ensure environment has enough CPU + Memory to run")
		minimumRequiredCPU := resource.MustParse("15")
		minimumRequiredMemory := resource.MustParse("48Gi")
		localNodeCap := getLocalNode(f).Status.Allocatable
		cpuCap := localNodeCap[v1.ResourceCPU]
		memCap := localNodeCap[v1.ResourceMemory]
		if cpuCap.Cmp(minimumRequiredCPU) == -1 {
			e2eskipper.Skipf("Skipping Node Performance Tests due to lack of CPU. Required %v is less than capacity %v.", minimumRequiredCPU, cpuCap)
		}
		if memCap.Cmp(minimumRequiredMemory) == -1 {
			e2eskipper.Skipf("Skipping Node Performance Tests due to lack of memory. Required %v is less than capacity %v.", minimumRequiredMemory, memCap)
		}
	})

	ginkgo.Context("Run node performance testing with pre-defined workloads", func() {
		ginkgo.BeforeEach(func() {
			wl = workloads.NodePerfWorkloads[0]
		})
		ginkgo.It("NAS parallel benchmark (NPB) suite - Integer Sort (IS) workload", func() {
			defer cleanup()
			runWorkload()
		})
	})
	ginkgo.Context("Run node performance testing with pre-defined workloads", func() {
		ginkgo.BeforeEach(func() {
			wl = workloads.NodePerfWorkloads[1]
		})
		ginkgo.It("NAS parallel benchmark (NPB) suite - Embarrassingly Parallel (EP) workload", func() {
			defer cleanup()
			runWorkload()
		})
	})
	ginkgo.Context("Run node performance testing with pre-defined workloads", func() {
		ginkgo.BeforeEach(func() {
			wl = workloads.NodePerfWorkloads[2]
		})
		ginkgo.It("TensorFlow workload", func() {
			defer cleanup()
			runWorkload()
		})
	})
})
