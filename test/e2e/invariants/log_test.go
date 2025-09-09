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

package invariants

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
	"testing/synctest"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/test/e2e/storage/podlogs"
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
		logs         []containerLogs
		check        logCheck
		podLogsError error
		expectFailed bool
		expectReport string
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
			check:        logCheck{dataRaces: true},
			expectFailed: true,
			expectReport: `#### kube-system/pod/container

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
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
			check:        logCheck{dataRaces: false},
			expectFailed: false,
			expectReport: ``,
		},
		"monitoring-errors": {
			logs:         []containerLogs{{"kube-system", "pod", "container", ``}},
			check:        logCheck{dataRaces: true},
			podLogsError: errors.New("fake pod logs error"),
			expectFailed: true,
			expectReport: `#### Errors

ERROR: log output collection failed for kube-system: fake pod logs error
ERROR: fake error: no log output
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
			check:        logCheck{dataRaces: true},
			expectFailed: true,
			expectReport: `#### kube-system/pod/container

- DATA RACE:
  
      Write at ...
      Goroutine created at:
        scheduleHandler()

- DATA RACE:
  
      Write at ...
      Goroutine created at:
        otherScheduleHandler()
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
			check:        logCheck{dataRaces: true},
			expectFailed: true,
			expectReport: `#### kube-system/pod/container

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
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
			check:        logCheck{dataRaces: true},
			expectFailed: true,
			expectReport: `#### kube-system/pod/container1

Okay.

#### kube-system/pod/container2

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
			check:        logCheck{dataRaces: true},
			expectFailed: true,
			expectReport: `#### kube-system/pod1/container

Okay.

#### kube-system/pod2/container

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ctx := context.Background()
				ctx, cancel := context.WithCancel(ctx)
				var objs []runtime.Object
				for _, name := range []string{"kube-system", "default", "custom"} {
					objs = append(objs, &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}})
				}
				client := fake.NewClientset(objs...)
				lc := newLogChecker(client, cancel, tc.check)
				podLogs := func(ctx context.Context, cs kubernetes.Interface, ns string, to podlogs.LogOutput) error {
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
								_, _ = writer.Write([]byte(line))
							}
							_ = writer.(io.Closer).Close()
						}
					}()

					return tc.podLogsError
				}
				lc.run(ctx, podLogs)
				synctest.Wait()
				actualFailed, actualReport := lc.stop()
				assert.Equal(t, tc.expectFailed, actualFailed, "check failed")
				assert.Equal(t, tc.expectReport, actualReport, "report")
			})
		})
	}
}
