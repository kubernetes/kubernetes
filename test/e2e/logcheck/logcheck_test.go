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
	for name, tc := range map[string]struct {
		logs          []containerLogs
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
			// TODO: if we get errors, should we consider data race detection as failed?
			expectStdout: `<log header>] "Watching" namespace="kube-system"
<log header>] "Log output collection failed" err="fake pod logs error" namespace="kube-system"
`,
		},
		"log-error": {
			logs:  []containerLogs{{"kube-system", "pod", "container", ``}},
			check: logCheck{dataRaces: true},
			// TODO: if we get errors, should we consider data race detection as failed?
			expectStdout: `<log header>] "Watching" namespace="kube-system"
<log header>] "PodLogs status" namespace="kube-system" msg="ERROR: fake error: no log output\n"
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
	} {
		tCtx := ktesting.Init(t)
		tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
			ctx, cancel := context.WithCancel(tCtx)
			var objs []runtime.Object
			for _, name := range []string{"kube-system", "default", "other-namespace", "yet-another-namespace"} {
				objs = append(objs, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}})
			}
			client := fake.NewClientset(objs...)
			tmpDir := tCtx.TempDir()
			lc, err := newLogChecker(client, cancel, tc.check, tmpDir)
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
							to.StatusWriter.Write([]byte("ERROR: fake error: no log output\n"))
							continue
						}
						writer := to.LogOpen(log.pod, log.container)
						if writer == nil {
							continue
						}
						for _, line := range strings.Split(log.output, "\n") {
							_, _ = writer.Write([]byte(line + "\n"))
						}
						_ = writer.(io.Closer).Close()
					}
				}()

				return nil
			}
			lc.start(ctx, startPodLogs)

			// Wait for goroutines to spin up.
			// lc.stop() alone is not enough because it races
			// with adding more background goroutines.
			synctest.Wait()

			actualFailure, actualStdout := lc.stop()
			actualStdout = logHeaderRE.ReplaceAllString(actualStdout, "<log header>]")
			assert.Equal(tCtx, tc.expectFailure, actualFailure, "failure message")
			assert.Equal(tCtx, tc.expectStdout, actualStdout, "report")
		})
	}
}

var logHeaderRE = regexp.MustCompile(`(?m)^.*logcheck.go:[[:digit:]]+\]`)
