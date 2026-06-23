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

package audit

import (
	"context"
	"errors"
	"strings"
	"testing"

	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestAuditEventTotalIsBeta(t *testing.T) {
	eventCounter.Reset()
	defer eventCounter.Reset()

	want := `
		# HELP apiserver_audit_event_total [BETA] Counter of audit events generated and sent to the audit backend.
		# TYPE apiserver_audit_event_total counter
		apiserver_audit_event_total 1
	`

	ObserveEvent(context.Background())

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "apiserver_audit_event_total"); err != nil {
		t.Fatal(err)
	}
}

func TestAuditErrorTotalIsBeta(t *testing.T) {
	errorCounter.Reset()
	defer errorCounter.Reset()

	want := `
		# HELP apiserver_audit_error_total [BETA] Counter of audit events that failed to be audited properly. Plugin identifies the plugin affected by the error.
		# TYPE apiserver_audit_error_total counter
		apiserver_audit_error_total{plugin="test-plugin"} 1
	`

	HandlePluginError("test-plugin", errors.New("audit failed"), &auditinternal.Event{})

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "apiserver_audit_error_total"); err != nil {
		t.Fatal(err)
	}
}

func TestAuditLevelTotalIsBeta(t *testing.T) {
	levelCounter.Reset()
	defer levelCounter.Reset()

	want := `
		# HELP apiserver_audit_level_total [BETA] Counter of policy levels for audit events (1 per request).
		# TYPE apiserver_audit_level_total counter
		apiserver_audit_level_total{level="Metadata"} 1
	`

	ObservePolicyLevel(context.Background(), auditinternal.LevelMetadata)

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), "apiserver_audit_level_total"); err != nil {
		t.Fatal(err)
	}
}
