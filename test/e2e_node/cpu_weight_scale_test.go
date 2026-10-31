//go:build linux

/*
Copyright The Kubernetes Authors.

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
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

// cpuWeightForShares mirrors the kubelet's cgroup v2 shares to cpu.weight
// conversion for a given shares ceiling.
func cpuWeightForShares(shares, maxShares uint64) uint64 {
	if shares >= maxShares {
		return 10000
	}
	return 1 + ((shares-2)*9999)/(maxShares-2)
}

// On cgroup v2 nodes with more than 256 CPUs the kubelet scales pod cgroup
// cpu.weight to the node's capacity. Pods that keep running across an upgrade
// from a kubelet without that scaling still carry the old weight, which the new
// kubelet would otherwise read back and report as a larger CPU request.
var _ = SIGDescribe("CPU weight scaling", framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("cpu-weight-scale")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should rescale cpu.weight of pods that survive a kubelet restart", func(ctx context.Context) {
		if !IsCgroup2UnifiedMode() {
			e2eskipper.Skipf("This test requires cgroups v2")
		}
		node := getLocalNode(ctx, f)
		capacity := node.Status.Capacity.Cpu().MilliValue()
		newMaxShares := uint64(capacity) * cm.SharesPerCPU / cm.MilliCPUToCPU
		if newMaxShares <= cm.MaxShares {
			e2eskipper.Skipf("This test requires more than %d CPUs, the node has %dm", cm.MaxShares/cm.SharesPerCPU, capacity)
		}

		ginkgo.By("Creating a running Burstable pod requesting 1 CPU")
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "cpu-weight-" + string(uuid.NewUUID())},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{
					Name:      "pause",
					Image:     imageutils.GetPauseImageName(),
					Resources: getResourceRequirements(getResourceList("1", "64Mi"), v1.ResourceList{}),
				}},
			},
		}
		pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
		weightFile := filepath.Join(memqosGetPodCgroupPath(pod, getCgroupDriver()), "cpu.weight")

		shares := cm.MilliCPUToShares(1000)
		newWeight := cpuWeightForShares(shares, newMaxShares)
		legacyWeight := cpuWeightForShares(shares, cm.MaxShares)
		gomega.Expect(readCPUWeight(weightFile)).To(gomega.Equal(newWeight), "pod created by this kubelet should use the node-capacity scale")

		ginkgo.By(fmt.Sprintf("Stopping the kubelet and restoring the pre-upgrade cpu.weight %d", legacyWeight))
		startKubelet := mustStopKubelet(ctx, f)
		framework.ExpectNoError(os.WriteFile(weightFile, []byte(strconv.FormatUint(legacyWeight, 10)), 0644))

		ginkgo.By("Restarting the kubelet over the running pod")
		startKubelet(ctx)

		ginkgo.By(fmt.Sprintf("Waiting for cpu.weight to be rescaled to %d", newWeight))
		gomega.Eventually(ctx, func() (uint64, error) {
			return readCPUWeight(weightFile)
		}).WithTimeout(2 * time.Minute).WithPolling(2 * time.Second).Should(gomega.Equal(newWeight))

		ginkgo.By("Checking that the pod was not restarted and does not report more CPU than it requests")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			if got, want := p.Status.ContainerStatuses[0].RestartCount, int32(0); got != want {
				return fmt.Errorf("container restarted %d times", got)
			}
			// status.resources is only published when pod-level resources are enabled.
			if p.Status.Resources == nil {
				return nil
			}
			if got := p.Status.Resources.Requests.Cpu().MilliValue(); got > 1000 {
				return fmt.Errorf("status.resources.requests.cpu = %dm, want at most 1000m", got)
			}
			return nil
		}).WithTimeout(2 * time.Minute).WithPolling(5 * time.Second).Should(gomega.Succeed())
	})
})

func readCPUWeight(path string) (uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}
