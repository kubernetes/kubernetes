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

package e2e_node

import (
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e_node/perf/workloads"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// makeNodePerfPod returns a pod with the information provided from the workload.
func makeNodePerfPod(w workloads.NodePerfWorkload) *corev1.Pod {
	return &corev1.Pod{
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
	Eventually(func() bool {
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		return len(nodeList.Items) == 1
	}, time.Minute, time.Second).Should(BeTrue())
}

// Serial because the test updates kubelet configuration.
// Slow by design.
var _ = SIGDescribe("Node Performance Testing [Serial] [Slow]", func() {
	f := framework.NewDefaultFramework("node-performance-testing")

	Context("Run node performance testing with pre-defined workloads", func() {
		It("run each pre-defined workload", func() {
			By("running the workloads")
			for _, workload := range workloads.NodePerfWorkloads {
				By("running the pre test exec from the workload")
				err := workload.PreTestExec()
				framework.ExpectNoError(err)

				By("restarting kubelet with required configuration")
				// Get the Kubelet config required for this workload.
				oldCfg, err := getCurrentKubeletConfig()
				framework.ExpectNoError(err)

				newCfg, err := workload.KubeletConfig(oldCfg)
				framework.ExpectNoError(err)
				// Set the Kubelet config required for this workload.
				setKubeletConfig(f, newCfg)

				By("running the workload and waiting for success")
				// Make the pod for the workload.
				pod := makeNodePerfPod(workload)

				// Create the pod.
				pod = f.PodClient().CreateSync(pod)
				// Wait for pod success.
				f.PodClient().WaitForSuccess(pod.Name, workload.Timeout())
				podLogs, err := framework.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
				framework.ExpectNoError(err)
				perf, err := workload.ExtractPerformanceFromLogs(podLogs)
				framework.ExpectNoError(err)
				framework.Logf("Time to complete workload %s: %v", workload.Name(), perf)

				// Delete the pod.
				gp := int64(0)
				delOpts := metav1.DeleteOptions{
					GracePeriodSeconds: &gp,
				}
				f.PodClient().DeleteSync(pod.Name, &delOpts, framework.DefaultPodDeletionTimeout)

				By("running the post test exec from the workload")
				err = workload.PostTestExec()
				framework.ExpectNoError(err)

				// Set the Kubelet config back to the old one.
				setKubeletConfig(f, oldCfg)
			}
		})
	})
})
