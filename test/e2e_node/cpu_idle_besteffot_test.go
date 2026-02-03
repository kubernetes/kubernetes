package e2enode

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	cgroup2CPUIdle = "cpu.idle"
)

var _ = SIGDescribe("CPU Idle for BestEffort QoS", framework.WithSerial(), feature.CPUIdleForBestEffortQoS, framework.WithFeatureGate(features.CPUIdleForBestEffortQoS), func() {
	f := framework.NewDefaultFramework("cpu-idle-besteffort-tests")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("With CPUIdleForBestEffortQoS feature gate enabled", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *config.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
			initialConfig.FeatureGates["CPUIdleForBestEffortQoS"] = true
		})
		ginkgo.BeforeEach(func() {
			// skip if cgroupv2 is disabled
			if !IsCgroup2UnifiedMode() {
				e2eskipper.Skip("Test requires cgroup v2")
			}
			//skip if kernel version < 5.4
			if !IsKernelVersionAvailable(5, 4) {
				e2eskipper.Skip("Test requires kernel version >= 5.4")
			}
		})
		ginkgo.It("should set cpu.idle=1 for BestEffort QoS cgroup", func(ctx context.Context) {
			// Get the BestEffort cgroup path
			bestEffortCgroupName := cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, bestEffortCgroup)
			bestEffortCgroupPath := toCgroupFsName(bestEffortCgroupName)
			cpuIdleFilePath := filepath.Join(cgroupBasePath, bestEffortCgroupPath, cgroup2CPUIdle)

			// Read and verify cpu.idle value
			gomega.Eventually(ctx, func() (string, error) {
				content, err := os.ReadFile(cpuIdleFilePath)
				if err != nil {
					return "", err
				}
				return strings.TrimSpace(string(content)), nil
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.Equal("1"),
				"Expected cpu.idle=1 for BestEffort QoS cgroup at %s", cpuIdleFilePath)
		})
	})

	ginkgo.Context("With CPUIdleForBestEffortQoS feature gate disabled", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = map[string]bool{}
			}
			initialConfig.FeatureGates["CPUIdleQoS"] = false
		})
		ginkgo.BeforeEach(func() {
			// Skip if not running cgroup v2
			if !IsCgroup2UnifiedMode() {
				e2eskipper.Skip("Test requires cgroup v2")
			}
		})

		ginkgo.It("cpu.idle value will be set to 0 for besteffort qos cgroup,when feature is disabled", func(ctx context.Context) {
			// Get the BestEffort cgroup path
			bestEffortCgroupName := cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup, bestEffortCgroup)
			bestEffortCgroupPath := toCgroupFsName(bestEffortCgroupName)
			cpuIdleFilePath := filepath.Join(cgroupBasePath, bestEffortCgroupPath, cgroup2CPUIdle)

			content, err := os.ReadFile(cpuIdleFilePath)
			framework.ExpectNoError(err, "Failed to get cpu.idle value from path %s", cpuIdleFilePath)
			cpuIdleValue := strings.TrimSpace(string(content))
			gomega.Expect(cpuIdleValue).To(gomega.Equal("0"),
				"Expected cpu.idle=0 for BestEffort QoS cgroup at %s when feature gate is disabled", cpuIdleFilePath)

		})
	})
})
