/*
Copyright 2026 The Kubernetes Authors.

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
	"os/exec"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	// systemOOMEventReason mirrors the reason used by pkg/kubelet/oom.
	systemOOMEventReason = "SystemOOM"
	// The kubelet records node events in the default namespace.
	nodeEventNamespace = "default"
)

/*
The test simulates system OOM kills by writing fake OOM killer lines into
/dev/kmsg. The kernel assigns the sequence number and timestamp of every
record written there, so the test cannot control event ordering. It verifies
that every simulated OOM is reported exactly once, also across a kubelet
restart, when the whole kernel log is replayed to the OOM watcher.
*/
var _ = SIGDescribe("OOMWatcher", framework.WithSerial(), framework.WithDisruptive(), func() {
	f := framework.NewDefaultFramework("oom")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("reports every system OOM once and does not replay them after a kubelet restart", ginkgo.Label("LinuxOnly"), func(ctx context.Context) {
		// A unique process name isolates this run from OOM events recorded
		// earlier on the node.
		processName := "e2e-oom-" + utilrand.String(8)
		pids := []int{900001, 900002, 900003}

		for _, pid := range pids {
			framework.ExpectNoError(insertOOMEvent(ctx, processName, pid))
		}
		waitForOOMEventCount(ctx, f, processName, len(pids))

		ginkgo.By("restarting the kubelet so it replays the kernel log")
		restartKubelet(ctx, true)
		waitForKubeletToStart(ctx, f)

		// The watcher reads /dev/kmsg from the beginning right after start,
		// so any duplicate would show up within this window.
		gomega.Consistently(ctx, func(ctx context.Context) (int, error) {
			return countOOMEvents(ctx, f, processName)
		}, 30*time.Second, 2*time.Second).Should(gomega.Equal(len(pids)), "already reported OOM events were reported again after the kubelet restart")

		ginkgo.By("verifying that new OOM kills are still reported after the restart")
		framework.ExpectNoError(insertOOMEvent(ctx, processName, 900004))
		waitForOOMEventCount(ctx, f, processName, len(pids)+1)
	})
})

// insertOOMEvent writes the kernel lines the OOM parser needs to recognise a
// system OOM kill. Each echo is a separate write, and each write to /dev/kmsg
// becomes exactly one kernel log record. The kernel rate-limits /dev/kmsg
// writes to 10 records per 5 seconds by default, so keep the total small.
func insertOOMEvent(ctx context.Context, processName string, pid int) error {
	lines := []string{
		fmt.Sprintf("%s invoked oom-killer: gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP), order=0, oom_score_adj=0", processName),
		fmt.Sprintf("Out of memory: Killed process %d (%s) total-vm:1658008kB, anon-rss:1647488kB, file-rss:1920kB, shmem-rss:0kB, UID:1000 pgtables:3284kB oom_score_adj:0", pid, processName),
	}
	script := ""
	for _, line := range lines {
		script += fmt.Sprintf("echo '%s' > /dev/kmsg\n", line)
	}
	out, err := exec.CommandContext(ctx, "sudo", "sh", "-c", script).CombinedOutput()
	if err != nil {
		return fmt.Errorf("writing OOM event to /dev/kmsg: %w, output: %s", err, out)
	}
	return nil
}

// countOOMEvents returns how many times a SystemOOM event for processName was
// recorded. The event recorder folds repeated identical events into one object
// and bumps its count, so the counts are summed.
func countOOMEvents(ctx context.Context, f *framework.Framework, processName string) (int, error) {
	events, err := f.ClientSet.CoreV1().Events(nodeEventNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return 0, err
	}
	total := 0
	for _, event := range events.Items {
		if event.Reason != systemOOMEventReason || !strings.Contains(event.Message, processName) {
			continue
		}
		total += max(int(event.Count), 1)
	}
	return total, nil
}

func waitForOOMEventCount(ctx context.Context, f *framework.Framework, processName string, expected int) {
	gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
		return countOOMEvents(ctx, f, processName)
	}, time.Minute, 2*time.Second).Should(gomega.Equal(expected))
}
