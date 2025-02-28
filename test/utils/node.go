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

package utils

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	hugepagesCapacityFile = "nr_hugepages"
	hugepagesDirPrefix    = "/sys/kernel/mm/hugepages/hugepages"

	hugepagesSize2M = 2048
	hugepagesSize1G = 1048576
)

var (
	resourceToSize = map[string]int{
		v1.ResourceHugePagesPrefix + "2Mi": hugepagesSize2M,
		v1.ResourceHugePagesPrefix + "1Gi": hugepagesSize1G,
	}
)

// GetNodeCondition extracts the provided condition from the given status and returns that.
// Returns nil and -1 if the condition is not present, and the index of the located condition.
func GetNodeCondition(status *v1.NodeStatus, conditionType v1.NodeConditionType) (int, *v1.NodeCondition) {
	if status == nil {
		return -1, nil
	}
	for i := range status.Conditions {
		if status.Conditions[i].Type == conditionType {
			return i, &status.Conditions[i]
		}
	}
	return -1, nil
}

func SetHugepages(ctx context.Context, hugepages map[string]int) {
	for hugepagesResource, count := range hugepages {
		size := resourceToSize[hugepagesResource]
		ginkgo.By(fmt.Sprintf("Verifying hugepages %d are supported", size))
		if !IsHugePageAvailable(size) {
			skipf("skipping test because hugepages of size %d not supported", size)
			return
		}

		ginkgo.By(fmt.Sprintf("Configuring the host to reserve %d of pre-allocated hugepages of size %d", count, size))
		gomega.Eventually(ctx, func() error {
			if err := ConfigureHugePages(size, count, nil); err != nil {
				return err
			}
			return nil
		}, 30*time.Second, framework.Poll).Should(gomega.BeNil())
	}
}

func IsHugePageAvailable(size int) bool {
	// e.g. /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
	hugepagesFile := fmt.Sprintf("/sys/kernel/mm/hugepages/hugepages-%dkB/nr_hugepages", size)
	if _, err := os.Stat(hugepagesFile); err != nil {
		framework.Logf("Hugepages file %s not found: %v", hugepagesFile, err)
		return false
	}
	return true
}

// configureHugePages attempts to allocate hugepages of the specified size
func ConfigureHugePages(hugepagesSize int, hugepagesCount int, numaNodeID *int) error {
	// Compact memory to make bigger contiguous blocks of memory available
	// before allocating huge pages.
	// https://www.kernel.org/doc/Documentation/sysctl/vm.txt
	if _, err := os.Stat("/proc/sys/vm/compact_memory"); err == nil {
		if err := exec.Command("/bin/sh", "-c", "echo 1 > /proc/sys/vm/compact_memory").Run(); err != nil {
			return err
		}
	}

	// e.g. hugepages/hugepages-2048kB/nr_hugepages
	hugepagesSuffix := fmt.Sprintf("hugepages/hugepages-%dkB/%s", hugepagesSize, hugepagesCapacityFile)

	// e.g. /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
	hugepagesFile := fmt.Sprintf("/sys/kernel/mm/%s", hugepagesSuffix)
	if numaNodeID != nil {
		// e.g. /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
		hugepagesFile = fmt.Sprintf("/sys/devices/system/node/node%d/%s", *numaNodeID, hugepagesSuffix)
	}

	// Reserve number of hugepages
	// e.g. /bin/sh -c "echo 5 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"
	command := fmt.Sprintf("echo %d > %s", hugepagesCount, hugepagesFile)
	if err := exec.Command("/bin/sh", "-c", command).Run(); err != nil {
		return err
	}

	// verify that the number of hugepages was updated
	// e.g. /bin/sh -c "cat /sys/kernel/mm/hugepages/hugepages-2048kB/vm.nr_hugepages"
	command = fmt.Sprintf("cat %s", hugepagesFile)
	outData, err := exec.Command("/bin/sh", "-c", command).Output()
	if err != nil {
		return err
	}

	numHugePages, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	if err != nil {
		return err
	}

	framework.Logf("Hugepages total is set to %v", numHugePages)
	if numHugePages == hugepagesCount {
		return nil
	}

	return fmt.Errorf("expected hugepages %v, but found %v", hugepagesCount, numHugePages)
}

// TODO(KevinTMtz) - Deduplicate from test/e2e_node/util.go:restartKubelet
func RestartKubelet(ctx context.Context, running bool) {
	kubeletServiceName := FindKubeletServiceName(running)
	// reset the kubelet service start-limit-hit
	stdout, err := exec.CommandContext(ctx, "sudo", "systemctl", "reset-failed", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to reset kubelet start-limit-hit with systemctl: %v, %s", err, string(stdout))

	stdout, err = exec.CommandContext(ctx, "sudo", "systemctl", "restart", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to restart kubelet with systemctl: %v, %s", err, string(stdout))
}

func FindKubeletServiceName(running bool) string {
	cmdLine := []string{
		"systemctl", "list-units", "*kubelet*",
	}
	if running {
		cmdLine = append(cmdLine, "--state=running")
	}
	stdout, err := exec.Command("sudo", cmdLine...).CombinedOutput()
	framework.ExpectNoError(err)
	regex := regexp.MustCompile("(kubelet-\\w+)")
	matches := regex.FindStringSubmatch(string(stdout))
	gomega.Expect(matches).ToNot(gomega.BeEmpty(), "Found more than one kubelet service running: %q", stdout)
	kubeletServiceName := matches[0]
	framework.Logf("Get running kubelet with systemctl: %v, %v", string(stdout), kubeletServiceName)
	return kubeletServiceName
}

func WaitForHugepages(ctx context.Context, f *framework.Framework, hugepages map[string]int) {
	ginkgo.By("Waiting for hugepages resource to become available on the local node")
	gomega.Eventually(ctx, func(ctx context.Context) error {
		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}

		for hugepagesResource, count := range hugepages {
			capacity, ok := node.Status.Capacity[v1.ResourceName(hugepagesResource)]
			if !ok {
				return fmt.Errorf("the node does not have the resource %s", hugepagesResource)
			}

			size, succeed := capacity.AsInt64()
			if !succeed {
				return fmt.Errorf("failed to convert quantity to int64")
			}

			expectedSize := count * resourceToSize[hugepagesResource] * 1024
			if size != int64(expectedSize) {
				return fmt.Errorf("the actual size %d is different from the expected one %d", size, expectedSize)
			}
		}
		return nil
	}, time.Minute, framework.Poll).Should(gomega.BeNil())
}

func ReleaseHugepages(ctx context.Context, hugepages map[string]int) {
	ginkgo.By("Releasing hugepages")
	gomega.Eventually(ctx, func() error {
		for hugepagesResource := range hugepages {
			command := fmt.Sprintf("echo 0 > %s-%dkB/%s", hugepagesDirPrefix, resourceToSize[hugepagesResource], hugepagesCapacityFile)
			if err := exec.Command("/bin/sh", "-c", command).Run(); err != nil {
				return err
			}
		}
		return nil
	}, 30*time.Second, framework.Poll).Should(gomega.BeNil())
}

// TODO(KevinTMtz) - Deduplicate from test/e2e/framework/skipper/skipper.go:Skipf
func skipf(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	ginkgo.Skip(msg, 2)

	panic("unreachable")
}
