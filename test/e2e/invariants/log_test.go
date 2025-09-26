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
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type containerLogs struct {
	podName, containerName, output string
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
		logs                []containerLogs
		check               logCheck
		expectFailed        bool
		expectNumLogEntries int
		expectReport        string
	}{
		"empty": {},
		"data-race": {
			logs: []containerLogs{{"pod", "container", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`}},
			check:        logCheck{dataRaces: true, errors: true},
			expectFailed: true,
			expectReport: `#### pod

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
`,
		},
		"disabled-data-race": {
			logs: []containerLogs{{"pod", "container", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`}},
			check:        logCheck{dataRaces: false, errors: true},
			expectFailed: false,
			expectReport: `#### pod

Okay.
`,
		},
		"two-data-races": {
			logs: []containerLogs{{"pod", "container", `==================
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
			check:        logCheck{dataRaces: true, errors: true},
			expectFailed: true,
			expectReport: `#### pod

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
			logs: []containerLogs{{"pod", "container", logHeader + `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
` + logTail}},
			check:               logCheck{dataRaces: true, errors: true},
			expectFailed:        true,
			expectNumLogEntries: 6,
			expectReport: `#### pod

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
`,
		},
		"two-containers": {
			logs: []containerLogs{
				{"pod", "container1", logHeader},
				{"pod", "container2", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`},
			},
			check:               logCheck{dataRaces: true, errors: true},
			expectFailed:        true,
			expectNumLogEntries: 3,
			expectReport: `#### pod container1

Okay.

#### pod container2

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
`,
		},
		"two-pods": {
			logs: []containerLogs{
				{"pod1", "container", logHeader},
				{"pod2", "container", `==================
WARNING: DATA RACE
Write at ...
Goroutine created at:
  scheduleHandler()
==================
`},
			},
			check:               logCheck{dataRaces: true, errors: true},
			expectFailed:        true,
			expectNumLogEntries: 3,
			expectReport: `#### pod1

Okay.

#### pod2

DATA RACE:

    Write at ...
    Goroutine created at:
      scheduleHandler()
`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			getLogs := func(ctx context.Context, namespace, podName, containerName string) (string, error) {
				for _, log := range tc.logs {
					if log.podName == podName && log.containerName == containerName {
						if log.output == "" {
							return "", errors.New("fake error: no log output")
						}
						return log.output, nil
					}
				}
				return "", errors.New("log output not found")
			}

			// We test with every container which is listed in the test case.
			// Retrieving their log output can be made to fail if it is empty.
			var pods []v1.Pod
			for _, log := range tc.logs {
				index := slices.IndexFunc(pods, func(pod v1.Pod) bool {
					return pod.Name == log.podName
				})
				if index < 0 {
					pods = append(pods, v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: log.podName}})
					index = len(pods) - 1
				}
				pods[index].Spec.Containers = append(pods[index].Spec.Containers, v1.Container{Name: log.containerName})
			}

			actualFailed, actualNumLogEntries, actualReport := checkLogs(ctx, getLogs, pods, tc.check)
			assert.Equal(t, tc.expectFailed, actualFailed, "check failed")
			assert.Equal(t, tc.expectNumLogEntries, actualNumLogEntries, "number of log entries")
			assert.Equal(t, tc.expectReport, actualReport, "report")
		})
	}
}
