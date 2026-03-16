/*
Copyright 2024 The Kubernetes Authors.

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

package audit

import (
	"context"
	"strings"
	"testing"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

type testError struct {
	msg string
}

func (e *testError) Error() string {
	return e.msg
}

func TestAuditMetrics(t *testing.T) {
	testCases := []struct {
		desc    string
		metrics []string
		update  func()
		want    string
	}{
		{
			desc:    "event_total counter",
			metrics: []string{"apiserver_audit_event_total"},
			update: func() {
				ObserveEvent(context.Background())
			},
			want: `
			# HELP apiserver_audit_event_total [BETA] Counter of audit events generated and sent to the audit backend.
			# TYPE apiserver_audit_event_total counter
			apiserver_audit_event_total 1
			`,
		},
		{
			desc:    "error_total counter",
			metrics: []string{"apiserver_audit_error_total"},
			update: func() {
				// HandlePluginError increments the counter based on the number of impacted events
				err := &testError{msg: "test error"}
				impactedEvent := &auditinternal.Event{}
				HandlePluginError("test-plugin", err, impactedEvent)
			},
			want: `
			# HELP apiserver_audit_error_total [BETA] Counter of audit events that failed to be audited properly. Plugin identifies the plugin affected by the error.
			# TYPE apiserver_audit_error_total counter
			apiserver_audit_error_total{plugin="test-plugin"} 1
			`,
		},
		{
			desc:    "level_total counter",
			metrics: []string{"apiserver_audit_level_total"},
			update: func() {
				ObservePolicyLevel(context.Background(), auditinternal.LevelRequest)
			},
			want: `
			# HELP apiserver_audit_level_total [BETA] Counter of policy levels for audit events (1 per request).
			# TYPE apiserver_audit_level_total counter
			apiserver_audit_level_total{level="Request"} 1
			`,
		},
	}

	for _, test := range testCases {
		t.Run(test.desc, func(t *testing.T) {
			eventCounter.Reset()
			errorCounter.Reset()
			levelCounter.Reset()
			test.update()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(test.want), test.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}
