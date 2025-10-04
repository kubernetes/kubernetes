/*
Copyright 2022 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"os/exec"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("OOMWatcher", func() {
	f := framework.NewDefaultFramework("oom")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("ignores already processed events [LinuxOnly]", func(ctx context.Context) {
		events, err := f.ClientSet.CoreV1().Events("default").List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)

		OOMEventOnTestStart := getOOMEventOnly(events.Items)
		framework.Logf("OOM Event count on test start: '%d'", len(OOMEventOnTestStart))

		offsets := []time.Duration{
			10 * time.Second,
			1 * time.Second,
			5 * time.Second,
			7 * time.Second,
			15 * time.Second,
		}

		for k, offset := range offsets {
			framework.Logf("Insert oom event with offset: '%d'", offset)
			err := insertOOMEvent(k, offset.Microseconds())
			framework.ExpectNoError(err)
		}

		events, err = f.ClientSet.CoreV1().Events("default").List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		OOMEventsAfterInsert := getOOMEventOnly(events.Items)
		framework.Logf("OOM Event count after insert: '%d'", len(OOMEventsAfterInsert))

		if len(OOMEventsAfterInsert) != (2 + len(OOMEventOnTestStart)) {
			framework.Failf("Expected %d OOM events, got %d", 2+len(OOMEventOnTestStart), len(OOMEventsAfterInsert))
		}
		restartKubelet(ctx, true)

		eventsAfterRestart, err := f.ClientSet.CoreV1().Events("default").List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		OOMEventsAfterKubeletRestart := getOOMEventOnly(eventsAfterRestart.Items)

		if len(OOMEventsAfterInsert) != len(OOMEventsAfterKubeletRestart) {
			framework.Failf("Expected %d OOM events, got %d after kubelet restart", len(OOMEventsAfterInsert), len(OOMEventsAfterKubeletRestart))
		}

	})
})

func insertOOMEvent(processID int, offset int64) error {
	cmd := fmt.Sprintf(`cat >> %s << EOF 
%s 
EOF`, "/dev/kmsg", buildOOMEvent(processID, offset))
	_, err := exec.Command("sudo", "sh", "-c", cmd).Output()
	var exError *exec.ExitError
	if errors.As(err, &exError) {
		framework.Logf("error %s", string(exError.Stderr))
	}

	if err != nil {
		return err
	}
	return nil
}

func buildOOMEvent(processID int, timeOffset int64) string {
	return fmt.Sprintf(`
4,1034,%d,-;sshd invoked oom-killer: gfp_mask=0x140cca(GFP_HIGHUSER_MOVABLE|__GFP_COMP), order=0, oom_score_adj=0
6,1143,%d,-;oom-kill:constraint=CONSTRAINT_NONE,nodemask=(null),cpuset=user.slice,mems_allowed=0,global_oom,task_memcg=/user.slice/user-1000.slice/session-4.scope,task=perl,pid=6118,uid=1000
3,1144,%d,-;Out of memory: Killed process %d (perl) total-vm:1658008kB, anon-rss:1647488kB, file-rss:1920kB, shmem-rss:0kB, UID:1000 pgtables:3284kB oom_score_adj:0`, timeOffset, timeOffset, timeOffset, processID)
}

func getOOMEventOnly(events []v1.Event) []v1.Event {
	result := []v1.Event{}
	for _, event := range events {
		if event.Reason == "SystemOOM" {
			result = append(result, event)
		}
	}
	return result
}
