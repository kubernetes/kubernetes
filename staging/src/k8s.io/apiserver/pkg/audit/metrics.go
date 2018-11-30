/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/klog"
)

const (
	subsystem = "apiserver_audit"
)

var (
	eventCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "event_total",
			Help:      "Counter of audit events generated and sent to the audit backend.",
		})
	errorCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "error_total",
			Help: "Counter of audit events that failed to be audited properly. " +
				"Plugin identifies the plugin affected by the error.",
		},
		[]string{"plugin"},
	)
	levelCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "level_total",
			Help:      "Counter of policy levels for audit events (1 per request).",
		},
		[]string{"level"},
	)

	ApiserverAuditDroppedCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: subsystem,
			Name:      "requests_rejected_total",
			Help: "Counter of apiserver requests rejected due to an error " +
				"in audit logging backend.",
		},
	)
)

func init() {
	prometheus.MustRegister(eventCounter)
	prometheus.MustRegister(errorCounter)
	prometheus.MustRegister(levelCounter)
	prometheus.MustRegister(ApiserverAuditDroppedCounter)
}

// ObserveEvent updates the relevant prometheus metrics for the generated audit event.
func ObserveEvent() {
	eventCounter.Inc()
}

// ObservePolicyLevel updates the relevant prometheus metrics with the audit level for a request.
func ObservePolicyLevel(level auditinternal.Level) {
	levelCounter.WithLabelValues(string(level)).Inc()
}

// HandlePluginError handles an error that occurred in an audit plugin. This method should only be
// used if the error may have prevented the audit event from being properly recorded. The events are
// logged to the debug log.
func HandlePluginError(plugin string, err error, impacted ...*auditinternal.Event) {
	// Count the error.
	errorCounter.WithLabelValues(plugin).Add(float64(len(impacted)))

	// Log the audit events to the debug log.
	msg := fmt.Sprintf("Error in audit plugin '%s' affecting %d audit events: %v\nImpacted events:\n",
		plugin, len(impacted), err)
	for _, ev := range impacted {
		msg = msg + EventString(ev) + "\n"
	}
	klog.Error(msg)
}
