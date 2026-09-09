/*
Copyright 2025 The Kubernetes Authors.

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

package logcheck

import (
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strings"
	"testing"
	"testing/synctest"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/test/e2e/storage/podlogs"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type containerLogs struct {
	namespace, pod, container, output string
}

const (
	logHeader = `E0901 16:50:24.914187       1 reflector.go:418] "The watchlist request ended with an error, falling back to the standard LIST/WATCH semantics because making progress is better than deadlocking" err="the server could not find the requested resource"
E0901 16:50:24.918970       1 reflector.go:203] "Failed to watch" err="failed to list *v1.PartialObjectMetadata: the server could not find the requested resource" logger="UnhandledError" reflector="k8s.io/client-go/metadata/metadatainformer/informer.go:138" type="*v1.PartialObjectMetadata"
I0901 16:55:24.615827       1 pathrecorder.go:243] healthz: "/healthz" satisfied by exact match
`
	logTail = `I0901 16:55:24.619018       1 httplog.go:134] "HTTP" verb="GET" URI="/healthz" latency="3.382408ms" userAgent="kube-probe/1.35+" audit-ID="" srcIP="127.0.0.1:34558" resp=200
E0901 17:10:42.624619       1 reflector.go:418] "The watchlist request ended with an error, falling back to the standard LIST/WATCH semantics because making progress is better than deadlocking" err="the server could not find the requested resource"
E0901 17:10:42.630037       1 reflector.go:203] "Failed to watch" err="failed to list *v1.PartialObjectMetadata: the server could not find the requested resource" logger="UnhandledError" reflector="k8s.io/client-go/metadata/metadatainformer/informer.go:138" type="*v1.PartialObjectMetadata"
`
)

func TestCheckLogs(t *testing.T) {
	nothing := regexpValue{regexp.MustCompile(`^$`)}

	for name, tc := range map[string]struct {
		logs          []containerLogs
		numWorkers    int
		kubeletLogs   map[string]string
		check         logCheck
		podLogsError  error
		expectFailure string
		expectStdout  string
	}{
		"empty": {},
		"data-race": {
			logs: []containerLogs{{"kube-system", "pod", "container", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`}},
			check: logCheck{dataRaces: true},
			expectFailure: `#### kube-system/pod/container

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
`,
			expectStdout: `<log header>] "Watching" namespace="kube-system"
<log header>] "Starting to check for data races" container="kube-system/pod/container"
<log header>] "Started new data race" container="kube-system/pod/container" count=1
<log header>] "Completed data race" container="kube-system/pod/container" count=1 dataRace=<
	Write at ...
	Goroutine created at:
	  scheduleHandler()
 >
<log header>] "Done checking for data races" container="kube-system/pod/container"
`,
		},
		"disabled-data-race": {
			logs: []containerLogs{{"kube-system", "pod", "container", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`}},
			check: logCheck{dataRaces: false},
		},
		"ignored-data-race-in-default-namespace": {
			logs: []containerLogs{{"default", "pod", "container", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`}},
			check: logCheck{dataRaces: true},
			expectStdout: `<log header>] "Watching" namespace="kube-system"
`,
		},
		"ignored-data-race-because-of-namespace-config": {
			logs: []containerLogs{{"kube-system", "pod", "container", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`}},
			check: logCheck{dataRaces: true, namespaces: regexpValue{regexp.MustCompile("other")}},
			expectStdout: `<log header>] "Watching" namespace="other-namespace"
<log header>] "Watching" namespace="yet-another-namespace"
`,
		},
		"start-error": {
			logs:         []containerLogs{{"kube-system", "pod", "container", ``}},
			check:        logCheck{dataRaces: true},
			podLogsError: errors.New("fake pod logs error"),
			expectStdout: `<log header>] "Watching" namespace="kube-system"
<log header>] "Log output collection failed" err="fake pod logs error" namespace="kube-system"
`,
			expectFailure: `Unexpected errors during log data collection (see stdout for full log):

    <log header>] "Log output collection failed" err="fake pod logs error" namespace="kube-system"
`,
		},
		"log-error": {
			logs:  []containerLogs{{"kube-system", "pod", "container", ``}},
			check: logCheck{dataRaces: true},
			expectStdout: `<log header>] "Watching" namespace="kube-system"
<log header>] "PodLogs status" namespace="kube-system" msg="ERROR: fake error: no log output"
`,
			expectFailure: `Unexpected errors during log data collection (see stdout for full log):

    <log header>] "PodLogs status" namespace="kube-system" msg="ERROR: fake error: no log output"
`,
		},
		"two-data-races": {
			logs: []containerLogs{{"kube-system", "pod", "container", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  otherScheduleHandler()
==================
`}},
			check: logCheck{dataRaces: true},
			expectFailure: `#### kube-system/pod/container

- DATA RACE:
  
      Write at ...
      Goroutine created at:
        scheduleHandler()

- DATA RACE:
  
      Write at ...
      Goroutine created at:
        otherScheduleHandler()
`,
			expectStdout: `<log header>] "Watching" namespace="kube-system"
<log header>] "Starting to check for data races" container="kube-system/pod/container"
<log header>] "Started new data race" container="kube-system/pod/container" count=1
<log header>] "Completed data race" container="kube-system/pod/container" count=1 dataRace=<
	Write at ...
	Goroutine created at:
	  scheduleHandler()
 >
<log header>] "Started new data race" container="kube-system/pod/container" count=2
<log header>] "Completed data race" container="kube-system/pod/container" count=2 dataRace=<
	Write at ...
	Goroutine created at:
	  otherScheduleHandler()
 >
<log header>] "Done checking for data races" container="kube-system/pod/container"
`,
		},
		"both": {
			logs: []containerLogs{{"kube-system", "pod", "container", logHeader + `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
` + logTail}},
			check: logCheck{dataRaces: true},
			expectFailure: `#### kube-system/pod/container

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
`,
			expectStdout: `<log header>] "Watching" namespace="kube-system"
<log header>] "Starting to check for data races" container="kube-system/pod/container"
<log header>] "Started new data race" container="kube-system/pod/container" count=1
<log header>] "Completed data race" container="kube-system/pod/container" count=1 dataRace=<
	Write at ...
	Goroutine created at:
	  scheduleHandler()
 >
<log header>] "Done checking for data races" container="kube-system/pod/container"
`,
		},
		"two-containers": {
			logs: []containerLogs{
				{"kube-system", "pod", "container1", logHeader},
				{"kube-system", "pod", "container2", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`},
			},
			check: logCheck{dataRaces: true},
			expectStdout: `<log header>] "Watching" namespace="kube-system"
<log header>] "Starting to check for data races" container="kube-system/pod/container1"
<log header>] "Done checking for data races" container="kube-system/pod/container1"
<log header>] "Starting to check for data races" container="kube-system/pod/container2"
<log header>] "Started new data race" container="kube-system/pod/container2" count=1
<log header>] "Completed data race" container="kube-system/pod/container2" count=1 dataRace=<
	Write at ...
	Goroutine created at:
	  scheduleHandler()
 >
<log header>] "Done checking for data races" container="kube-system/pod/container2"

#### kube-system/pod/container1

Okay.
`,
			expectFailure: `#### kube-system/pod/container2

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
`,
		},
		"two-pods": {
			logs: []containerLogs{
				{"kube-system", "pod1", "container", logHeader},
				{"kube-system", "pod2", "container", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`},
			},
			check: logCheck{dataRaces: true},
			expectStdout: `<log header>] "Watching" namespace="kube-system"
<log header>] "Starting to check for data races" container="kube-system/pod1/container"
<log header>] "Done checking for data races" container="kube-system/pod1/container"
<log header>] "Starting to check for data races" container="kube-system/pod2/container"
<log header>] "Started new data race" container="kube-system/pod2/container" count=1
<log header>] "Completed data race" container="kube-system/pod2/container" count=1 dataRace=<
	Write at ...
	Goroutine created at:
	  scheduleHandler()
 >
<log header>] "Done checking for data races" container="kube-system/pod2/container"

#### kube-system/pod1/container

Okay.
`,
			expectFailure: `#### kube-system/pod2/container

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
`,
		},
		"kubelet-all-nodes": {
			numWorkers: 3,
			kubeletLogs: map[string]string{
				"worker1": `Some other output....
... more output.
`,
				"worker2": `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  handlePod2()
==================
`,
			},
			check: logCheck{dataRaces: true, namespaces: nothing},
			expectStdout: `<log header>] "Watching" node="worker0"
<log header>] "Watching" node="worker1"
<log header>] "Watching" node="worker2"
<log header>] "Started new data race" container="kubelet/worker2" count=1
<log header>] "Completed data race" container="kubelet/worker2" count=1 dataRace=<
	Write at ...
	Goroutine created at:
	  handlePod2()
 >

#### kubelet/worker1

Okay.
`,
			expectFailure: `#### kubelet/worker2

DATA RACE:

    Write at ...
    Goroutine created at:
      handlePod2()
`,
		},
		"kubelet-no-nodes": {
			numWorkers: 3,
			kubeletLogs: map[string]string{
				"worker1": `Some other output....
... more output.
`,
				"worker2": `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  handlePod2()
==================
`,
			},
			check: logCheck{dataRaces: true, namespaces: nothing, nodes: nothing},
		},
		"kubelet-long": {
			numWorkers: 3,
			kubeletLogs: map[string]string{
				// This is a real example. The race detector seems to replicate stack entries,
				// so we should better truncate in the middle to keep failure messages short.
				"worker2": `==================
WARNING: DATA RACE
Write at 0x00c0010def18 by goroutine 285:
  k8s.io/kubernetes/pkg/kubelet/status.normalizeStatus.func1()
      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1193 +0x1ee
  k8s.io/kubernetes/pkg/kubelet/status.normalizeStatus()
      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1209 +0x175
  k8s.io/kubernetes/pkg/kubelet/status.(*manager).updateStatusInternal()
      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:838 +0x9dc
  k8s.io/kubernetes/pkg/kubelet/status.(*manager).SetContainerReadiness()
      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:501 +0x20cb
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2650 +0x25a5
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).updateCache()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:83 +0x3e
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).Run()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:1896 +0xfea
  k8s.io/kubernetes/cmd/kubelet/app.startKubelet.gowrap1()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:1264 +0x50
Previous read at 0x00c0010def18 by goroutine 537:
  k8s.io/apimachinery/pkg/apis/meta/v1.(*Time).MarshalJSON()
      <autogenerated>:1 +0x44
  encoding/json.marshalerEncoder()
      encoding/json/encode.go:483 +0x13c
  encoding/json.structEncoder.encode()
      encoding/json/encode.go:758 +0x3c7
  encoding/json.structEncoder.encode-fm()
      <autogenerated>:1 +0xe4
  encoding/json.structEncoder.encode()
      encoding/json/encode.go:758 +0x3c7
  encoding/json.structEncoder.encode-fm()
      <autogenerated>:1 +0xe4
  encoding/json.(*encodeState).reflectValue()
      encoding/json/encode.go:367 +0x83
  encoding/json.(*encodeState).marshal()
      encoding/json/encode.go:343 +0xdb
  encoding/json.Marshal()
      encoding/json/encode.go:209 +0x11e
  k8s.io/kubernetes/pkg/util/pod.preparePatchBytesForPodStatus()
      k8s.io/kubernetes/pkg/util/pod/pod.go:58 +0x2d3
  k8s.io/kubernetes/pkg/util/pod.PatchPodStatus()
      k8s.io/kubernetes/pkg/util/pod/pod.go:35 +0x12b
  k8s.io/kubernetes/pkg/kubelet/status.(*manager).syncPod()
      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1064 +0xada
  k8s.io/kubernetes/pkg/kubelet/status.(*manager).syncBatch()
      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1025 +0x199
  k8s.io/kubernetes/pkg/kubelet/status.(*manager).Start.func1()
      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:260 +0x1a4
  k8s.io/apimachinery/pkg/util/wait.BackoffUntil.func1()
      k8s.io/apimachinery/pkg/util/wait/backoff.go:233 +0x2e
  k8s.io/apimachinery/pkg/util/wait.BackoffUntilWithContext.func1()
      k8s.io/apimachinery/pkg/util/wait/backoff.go:255 +0x98
  k8s.io/apimachinery/pkg/util/wait.BackoffUntilWithContext()
      k8s.io/apimachinery/pkg/util/wait/backoff.go:256 +0xed
  k8s.io/apimachinery/pkg/util/wait.BackoffUntil()
      k8s.io/apimachinery/pkg/util/wait/backoff.go:233 +0x8a
  k8s.io/apimachinery/pkg/util/wait.JitterUntil()
      k8s.io/apimachinery/pkg/util/wait/backoff.go:210 +0xfb
  k8s.io/apimachinery/pkg/util/wait.Until()
      k8s.io/apimachinery/pkg/util/wait/backoff.go:163 +0x50
  k8s.io/apimachinery/pkg/util/wait.Forever()
      k8s.io/apimachinery/pkg/util/wait/wait.go:80 +0x2a
  k8s.io/kubernetes/pkg/kubelet/status.(*manager).Start.gowrap1()
      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:255 +0x17
Goroutine 285 (running) created at:
  k8s.io/kubernetes/cmd/kubelet/app.startKubelet()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:1264 +0x14e
  k8s.io/kubernetes/cmd/kubelet/app.RunKubelet()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:1256 +0x7d7
  k8s.io/kubernetes/cmd/kubelet/app.createAndInitKubelet()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:1286 +0x5fd
  k8s.io/kubernetes/cmd/kubelet/app.RunKubelet()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:1235 +0x588
  k8s.io/kubernetes/cmd/kubelet/app.run()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:901 +0x3c3e
  k8s.io/kubernetes/cmd/kubelet/app.getCgroupDriverFromCRI()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:1382 +0x116
  k8s.io/cri-client/pkg.(*remoteRuntimeService).RuntimeConfig()
      k8s.io/cri-client/pkg/remote_runtime.go:926 +0x144
  k8s.io/kubernetes/cmd/kubelet/app.getCgroupDriverFromCRI()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:1382 +0x116
  k8s.io/cri-client/pkg.(*remoteRuntimeService).RuntimeConfig()
      k8s.io/cri-client/pkg/remote_runtime.go:926 +0x144
  k8s.io/kubernetes/cmd/kubelet/app.getCgroupDriverFromCRI()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:1382 +0x116
  k8s.io/kubernetes/cmd/kubelet/app.run()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:732 +0x10f9
  k8s.io/kubernetes/pkg/kubelet.PreInitRuntimeService()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:411 +0x384
  k8s.io/cri-client/pkg.NewRemoteRuntimeService()
      k8s.io/cri-client/pkg/remote_runtime.go:134 +0x118a
  k8s.io/kubernetes/pkg/kubelet.PreInitRuntimeService()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:408 +0x28f
  k8s.io/kubernetes/cmd/kubelet/app.run()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:727 +0x10cb
  k8s.io/kubernetes/cmd/kubelet/app.Run()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:532 +0x4b9
  k8s.io/kubernetes/cmd/kubelet/app.NewKubeletCommand.func1()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:295 +0x1837
  github.com/spf13/cobra.(*Command).execute()
      github.com/spf13/cobra@v1.10.0/command.go:1015 +0x113b
  github.com/spf13/cobra.(*Command).ExecuteC()
      github.com/spf13/cobra@v1.10.0/command.go:1148 +0x797
  github.com/spf13/cobra.(*Command).Execute()
      github.com/spf13/cobra@v1.10.0/command.go:1071 +0x4d0
  k8s.io/component-base/cli.run()
      k8s.io/component-base/cli/run.go:146 +0x4d1
  k8s.io/component-base/cli.Run()
      k8s.io/component-base/cli/run.go:44 +0x3b
  main.main()
      k8s.io/kubernetes/cmd/kubelet/kubelet.go:56 +0x2f
Goroutine 537 (running) created at:
  k8s.io/kubernetes/pkg/kubelet/status.(*manager).Start()
      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:255 +0x27e
  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).Run()
      k8s.io/kubernetes/pkg/kubelet/kubelet.go:1877 +0xdde
  k8s.io/kubernetes/cmd/kubelet/app.startKubelet.gowrap1()
      k8s.io/kubernetes/cmd/kubelet/app/server.go:1264 +0x50
==================
`,
			},
			check: logCheck{dataRaces: true, namespaces: nothing},
			expectStdout: `<log header>] "Watching" node="worker0"
<log header>] "Watching" node="worker1"
<log header>] "Watching" node="worker2"
<log header>] "Started new data race" container="kubelet/worker2" count=1
<log header>] "Completed data race" container="kubelet/worker2" count=1 dataRace=<
	Write at 0x00c0010def18 by goroutine 285:
	  k8s.io/kubernetes/pkg/kubelet/status.normalizeStatus.func1()
	      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1193 +0x1ee
	  k8s.io/kubernetes/pkg/kubelet/status.normalizeStatus()
	      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1209 +0x175
	  k8s.io/kubernetes/pkg/kubelet/status.(*manager).updateStatusInternal()
	      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:838 +0x9dc
	  k8s.io/kubernetes/pkg/kubelet/status.(*manager).SetContainerReadiness()
	      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:501 +0x20cb
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2650 +0x25a5
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).updateCache()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:83 +0x3e
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet/container.(*runtimeCache).ForceUpdateIfOlder()
	      k8s.io/kubernetes/pkg/kubelet/container/runtime_cache.go:77 +0x130
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
	      k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).Run()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:1896 +0xfea
	  k8s.io/kubernetes/cmd/kubelet/app.startKubelet.gowrap1()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:1264 +0x50
	Previous read at 0x00c0010def18 by goroutine 537:
	  k8s.io/apimachinery/pkg/apis/meta/v1.(*Time).MarshalJSON()
	      <autogenerated>:1 +0x44
	  encoding/json.marshalerEncoder()
	      encoding/json/encode.go:483 +0x13c
	  encoding/json.structEncoder.encode()
	      encoding/json/encode.go:758 +0x3c7
	  encoding/json.structEncoder.encode-fm()
	      <autogenerated>:1 +0xe4
	  encoding/json.structEncoder.encode()
	      encoding/json/encode.go:758 +0x3c7
	  encoding/json.structEncoder.encode-fm()
	      <autogenerated>:1 +0xe4
	  encoding/json.(*encodeState).reflectValue()
	      encoding/json/encode.go:367 +0x83
	  encoding/json.(*encodeState).marshal()
	      encoding/json/encode.go:343 +0xdb
	  encoding/json.Marshal()
	      encoding/json/encode.go:209 +0x11e
	  k8s.io/kubernetes/pkg/util/pod.preparePatchBytesForPodStatus()
	      k8s.io/kubernetes/pkg/util/pod/pod.go:58 +0x2d3
	  k8s.io/kubernetes/pkg/util/pod.PatchPodStatus()
	      k8s.io/kubernetes/pkg/util/pod/pod.go:35 +0x12b
	  k8s.io/kubernetes/pkg/kubelet/status.(*manager).syncPod()
	      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1064 +0xada
	  k8s.io/kubernetes/pkg/kubelet/status.(*manager).syncBatch()
	      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1025 +0x199
	  k8s.io/kubernetes/pkg/kubelet/status.(*manager).Start.func1()
	      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:260 +0x1a4
	  k8s.io/apimachinery/pkg/util/wait.BackoffUntil.func1()
	      k8s.io/apimachinery/pkg/util/wait/backoff.go:233 +0x2e
	  k8s.io/apimachinery/pkg/util/wait.BackoffUntilWithContext.func1()
	      k8s.io/apimachinery/pkg/util/wait/backoff.go:255 +0x98
	  k8s.io/apimachinery/pkg/util/wait.BackoffUntilWithContext()
	      k8s.io/apimachinery/pkg/util/wait/backoff.go:256 +0xed
	  k8s.io/apimachinery/pkg/util/wait.BackoffUntil()
	      k8s.io/apimachinery/pkg/util/wait/backoff.go:233 +0x8a
	  k8s.io/apimachinery/pkg/util/wait.JitterUntil()
	      k8s.io/apimachinery/pkg/util/wait/backoff.go:210 +0xfb
	  k8s.io/apimachinery/pkg/util/wait.Until()
	      k8s.io/apimachinery/pkg/util/wait/backoff.go:163 +0x50
	  k8s.io/apimachinery/pkg/util/wait.Forever()
	      k8s.io/apimachinery/pkg/util/wait/wait.go:80 +0x2a
	  k8s.io/kubernetes/pkg/kubelet/status.(*manager).Start.gowrap1()
	      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:255 +0x17
	Goroutine 285 (running) created at:
	  k8s.io/kubernetes/cmd/kubelet/app.startKubelet()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:1264 +0x14e
	  k8s.io/kubernetes/cmd/kubelet/app.RunKubelet()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:1256 +0x7d7
	  k8s.io/kubernetes/cmd/kubelet/app.createAndInitKubelet()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:1286 +0x5fd
	  k8s.io/kubernetes/cmd/kubelet/app.RunKubelet()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:1235 +0x588
	  k8s.io/kubernetes/cmd/kubelet/app.run()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:901 +0x3c3e
	  k8s.io/kubernetes/cmd/kubelet/app.getCgroupDriverFromCRI()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:1382 +0x116
	  k8s.io/cri-client/pkg.(*remoteRuntimeService).RuntimeConfig()
	      k8s.io/cri-client/pkg/remote_runtime.go:926 +0x144
	  k8s.io/kubernetes/cmd/kubelet/app.getCgroupDriverFromCRI()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:1382 +0x116
	  k8s.io/cri-client/pkg.(*remoteRuntimeService).RuntimeConfig()
	      k8s.io/cri-client/pkg/remote_runtime.go:926 +0x144
	  k8s.io/kubernetes/cmd/kubelet/app.getCgroupDriverFromCRI()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:1382 +0x116
	  k8s.io/kubernetes/cmd/kubelet/app.run()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:732 +0x10f9
	  k8s.io/kubernetes/pkg/kubelet.PreInitRuntimeService()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:411 +0x384
	  k8s.io/cri-client/pkg.NewRemoteRuntimeService()
	      k8s.io/cri-client/pkg/remote_runtime.go:134 +0x118a
	  k8s.io/kubernetes/pkg/kubelet.PreInitRuntimeService()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:408 +0x28f
	  k8s.io/kubernetes/cmd/kubelet/app.run()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:727 +0x10cb
	  k8s.io/kubernetes/cmd/kubelet/app.Run()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:532 +0x4b9
	  k8s.io/kubernetes/cmd/kubelet/app.NewKubeletCommand.func1()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:295 +0x1837
	  github.com/spf13/cobra.(*Command).execute()
	      github.com/spf13/cobra@v1.10.0/command.go:1015 +0x113b
	  github.com/spf13/cobra.(*Command).ExecuteC()
	      github.com/spf13/cobra@v1.10.0/command.go:1148 +0x797
	  github.com/spf13/cobra.(*Command).Execute()
	      github.com/spf13/cobra@v1.10.0/command.go:1071 +0x4d0
	  k8s.io/component-base/cli.run()
	      k8s.io/component-base/cli/run.go:146 +0x4d1
	  k8s.io/component-base/cli.Run()
	      k8s.io/component-base/cli/run.go:44 +0x3b
	  main.main()
	      k8s.io/kubernetes/cmd/kubelet/kubelet.go:56 +0x2f
	Goroutine 537 (running) created at:
	  k8s.io/kubernetes/pkg/kubelet/status.(*manager).Start()
	      k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:255 +0x27e
	  k8s.io/kubernetes/pkg/kubelet.(*Kubelet).Run()
	      k8s.io/kubernetes/pkg/kubelet/kubelet.go:1877 +0xdde
	  k8s.io/kubernetes/cmd/kubelet/app.startKubelet.gowrap1()
	      k8s.io/kubernetes/cmd/kubelet/app/server.go:1264 +0x50
 >
`,
			expectFailure: `#### kubelet/worker2

DATA RACE:

    Write at 0x00c0010def18 by goroutine 285:
      k8s.io/kubernetes/pkg/kubelet/status.normalizeStatus.func1()
          k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1193 +0x1ee
      k8s.io/kubernetes/pkg/kubelet/status.normalizeStatus()
          k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:1209 +0x175
      k8s.io/kubernetes/pkg/kubelet/status.(*manager).updateStatusInternal()
          k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:838 +0x9dc
      k8s.io/kubernetes/pkg/kubelet/status.(*manager).SetContainerReadiness()
          k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:501 +0x20cb
      k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
          k8s.io/kubernetes/pkg/kubelet/kubelet.go:2650 +0x25a5
      ...
      k8s.io/kubernetes/pkg/kubelet.(*Kubelet).HandlePodCleanups()
          k8s.io/kubernetes/pkg/kubelet/kubelet_pods.go:1263 +0x66f
      k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoopIteration()
          k8s.io/kubernetes/pkg/kubelet/kubelet.go:2690 +0x29b2
      k8s.io/kubernetes/pkg/kubelet.(*Kubelet).syncLoop()
          k8s.io/kubernetes/pkg/kubelet/kubelet.go:2542 +0x51d
      k8s.io/kubernetes/pkg/kubelet.(*Kubelet).Run()
          k8s.io/kubernetes/pkg/kubelet/kubelet.go:1896 +0xfea
      k8s.io/kubernetes/cmd/kubelet/app.startKubelet.gowrap1()
          k8s.io/kubernetes/cmd/kubelet/app/server.go:1264 +0x50
    Previous read at 0x00c0010def18 by goroutine 537:
      k8s.io/apimachinery/pkg/apis/meta/v1.(*Time).MarshalJSON()
          <autogenerated>:1 +0x44
      encoding/json.marshalerEncoder()
          encoding/json/encode.go:483 +0x13c
      encoding/json.structEncoder.encode()
          encoding/json/encode.go:758 +0x3c7
      encoding/json.structEncoder.encode-fm()
          <autogenerated>:1 +0xe4
      encoding/json.structEncoder.encode()
          encoding/json/encode.go:758 +0x3c7
      ...
      k8s.io/apimachinery/pkg/util/wait.BackoffUntil()
          k8s.io/apimachinery/pkg/util/wait/backoff.go:233 +0x8a
      k8s.io/apimachinery/pkg/util/wait.JitterUntil()
          k8s.io/apimachinery/pkg/util/wait/backoff.go:210 +0xfb
      k8s.io/apimachinery/pkg/util/wait.Until()
          k8s.io/apimachinery/pkg/util/wait/backoff.go:163 +0x50
      k8s.io/apimachinery/pkg/util/wait.Forever()
          k8s.io/apimachinery/pkg/util/wait/wait.go:80 +0x2a
      k8s.io/kubernetes/pkg/kubelet/status.(*manager).Start.gowrap1()
          k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:255 +0x17
    Goroutine 285 (running) created at:
      k8s.io/kubernetes/cmd/kubelet/app.startKubelet()
          k8s.io/kubernetes/cmd/kubelet/app/server.go:1264 +0x14e
      k8s.io/kubernetes/cmd/kubelet/app.RunKubelet()
          k8s.io/kubernetes/cmd/kubelet/app/server.go:1256 +0x7d7
      k8s.io/kubernetes/cmd/kubelet/app.createAndInitKubelet()
          k8s.io/kubernetes/cmd/kubelet/app/server.go:1286 +0x5fd
      k8s.io/kubernetes/cmd/kubelet/app.RunKubelet()
          k8s.io/kubernetes/cmd/kubelet/app/server.go:1235 +0x588
      k8s.io/kubernetes/cmd/kubelet/app.run()
          k8s.io/kubernetes/cmd/kubelet/app/server.go:901 +0x3c3e
      ...
      github.com/spf13/cobra.(*Command).ExecuteC()
          github.com/spf13/cobra@v1.10.0/command.go:1148 +0x797
      github.com/spf13/cobra.(*Command).Execute()
          github.com/spf13/cobra@v1.10.0/command.go:1071 +0x4d0
      k8s.io/component-base/cli.run()
          k8s.io/component-base/cli/run.go:146 +0x4d1
      k8s.io/component-base/cli.Run()
          k8s.io/component-base/cli/run.go:44 +0x3b
      main.main()
          k8s.io/kubernetes/cmd/kubelet/kubelet.go:56 +0x2f
    Goroutine 537 (running) created at:
      k8s.io/kubernetes/pkg/kubelet/status.(*manager).Start()
          k8s.io/kubernetes/pkg/kubelet/status/status_manager.go:255 +0x27e
      k8s.io/kubernetes/pkg/kubelet.(*Kubelet).Run()
          k8s.io/kubernetes/pkg/kubelet/kubelet.go:1877 +0xdde
      k8s.io/kubernetes/cmd/kubelet/app.startKubelet.gowrap1()
          k8s.io/kubernetes/cmd/kubelet/app/server.go:1264 +0x50
`,
		},
	} {
		tCtx := ktesting.Init(t)
		tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
			var objs []runtime.Object
			for _, name := range []string{"kube-system", "default", "other-namespace", "yet-another-namespace"} {
				objs = append(objs, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}})
			}
			for i := range tc.numWorkers {
				objs = append(objs, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("worker%d", i)}})
			}
			client := fake.NewClientset(objs...)
			tmpDir := tCtx.TempDir()
			ctx, lc, err := newLogChecker(tCtx, client, tc.check, tmpDir)
			require.NoError(tCtx, err)
			startPodLogs := func(ctx context.Context, cs kubernetes.Interface, ns string, to podlogs.LogOutput) error {
				if tc.podLogsError != nil {
					return tc.podLogsError
				}

				go func() {
					for _, log := range tc.logs {
						if log.namespace != ns {
							continue
						}
						if log.output == "" {
							_, _ = to.StatusWriter.Write([]byte("ERROR: fake error: no log output\n"))
							continue
						}
						writer := to.LogOpen(log.pod, log.container)
						if writer == nil {
							continue
						}
						for _, line := range strings.Split(log.output, "\n") {
							_, _ = writer.Write([]byte(line + "\n"))
						}
						if closer, ok := writer.(io.Closer); ok {
							_ = closer.Close()
						}
					}
				}()

				return nil
			}
			startNodeLog := func(ctx context.Context, cs kubernetes.Interface, wg *waitGroup, nodeName string) io.Reader {
				return strings.NewReader(tc.kubeletLogs[nodeName])
			}
			lc.start(ctx, startPodLogs, startNodeLog)

			// Wait for goroutines to spin up.
			// lc.stop() alone is not enough because it races
			// with adding more background goroutines.
			synctest.Wait()

			actualFailure, actualStdout := lc.stop(tCtx.Logger())
			actualStdout = logHeaderRE.ReplaceAllString(actualStdout, "$1<log header>]")
			assert.Equal(tCtx, tc.expectStdout, actualStdout, "report")
			actualFailure = logHeaderRE.ReplaceAllString(actualFailure, "$1<log header>]")
			assert.Equal(tCtx, tc.expectFailure, actualFailure, "failure message")
		})
	}
}

var logHeaderRE = regexp.MustCompile(`(?m)^(\s*).*logcheck.go:[[:digit:]]+\]`)
