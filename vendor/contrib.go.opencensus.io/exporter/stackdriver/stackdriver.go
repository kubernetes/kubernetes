// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package stackdriver contains the OpenCensus exporters for
// Stackdriver Monitoring and Stackdriver Tracing.
//
// This exporter can be used to send metrics to Stackdriver Monitoring and traces
// to Stackdriver trace.
//
// The package uses Application Default Credentials to authenticate by default.
// See: https://developers.google.com/identity/protocols/application-default-credentials
//
// Alternatively, pass the authentication options in both the MonitoringClientOptions
// and the TraceClientOptions fields of Options.
//
// Stackdriver Monitoring
//
// This exporter support exporting OpenCensus views to Stackdriver Monitoring.
// Each registered view becomes a metric in Stackdriver Monitoring, with the
// tags becoming labels.
//
// The aggregation function determines the metric kind: LastValue aggregations
// generate Gauge metrics and all other aggregations generate Cumulative metrics.
//
// In order to be able to push your stats to Stackdriver Monitoring, you must:
//
//   1. Create a Cloud project: https://support.google.com/cloud/answer/6251787?hl=en
//   2. Enable billing: https://support.google.com/cloud/answer/6288653#new-billing
//   3. Enable the Stackdriver Monitoring API: https://console.cloud.google.com/apis/dashboard
//
// These steps enable the API but don't require that your app is hosted on Google Cloud Platform.
//
// Stackdriver Trace
//
// This exporter supports exporting Trace Spans to Stackdriver Trace. It also
// supports the Google "Cloud Trace" propagation format header.
package stackdriver

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	traceapi "cloud.google.com/go/trace/apiv2"
	"contrib.go.opencensus.io/exporter/stackdriver/monitoredresource"
	"go.opencensus.io/stats/view"
	"go.opencensus.io/trace"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/option"
	monitoredrespb "google.golang.org/genproto/googleapis/api/monitoredres"
)

// Options contains options for configuring the exporter.
type Options struct {
	// ProjectID is the identifier of the Stackdriver
	// project the user is uploading the stats data to.
	// If not set, this will default to your "Application Default Credentials".
	// For details see: https://developers.google.com/accounts/docs/application-default-credentials
	ProjectID string

	// OnError is the hook to be called when there is
	// an error uploading the stats or tracing data.
	// If no custom hook is set, errors are logged.
	// Optional.
	OnError func(err error)

	// MonitoringClientOptions are additional options to be passed
	// to the underlying Stackdriver Monitoring API client.
	// Optional.
	MonitoringClientOptions []option.ClientOption

	// TraceClientOptions are additional options to be passed
	// to the underlying Stackdriver Trace API client.
	// Optional.
	TraceClientOptions []option.ClientOption

	// BundleDelayThreshold determines the max amount of time
	// the exporter can wait before uploading view data to
	// the backend.
	// Optional.
	BundleDelayThreshold time.Duration

	// BundleCountThreshold determines how many view data events
	// can be buffered before batch uploading them to the backend.
	// Optional.
	BundleCountThreshold int

	// Resource sets the MonitoredResource against which all views will be
	// recorded by this exporter.
	//
	// All Stackdriver metrics created by this exporter are custom metrics,
	// so only a limited number of MonitoredResource types are supported, see:
	// https://cloud.google.com/monitoring/custom-metrics/creating-metrics#which-resource
	//
	// An important consideration when setting the Resource here is that
	// Stackdriver Monitoring only allows a single writer per
	// TimeSeries, see: https://cloud.google.com/monitoring/api/v3/metrics-details#intro-time-series
	// A TimeSeries is uniquely defined by the metric type name
	// (constructed from the view name and the MetricPrefix), the Resource field,
	// and the set of label key/value pairs (in OpenCensus terminology: tag).
	//
	// If no custom Resource is set, a default MonitoredResource
	// with type global and no resource labels will be used. If you explicitly
	// set this field, you may also want to set custom DefaultMonitoringLabels.
	//
	// Deprecated: Use MonitoredResource instead.
	Resource *monitoredrespb.MonitoredResource

	// MonitoredResource sets the MonitoredResource against which all views will be
	// recorded by this exporter.
	//
	// All Stackdriver metrics created by this exporter are custom metrics,
	// so only a limited number of MonitoredResource types are supported, see:
	// https://cloud.google.com/monitoring/custom-metrics/creating-metrics#which-resource
	//
	// An important consideration when setting the MonitoredResource here is that
	// Stackdriver Monitoring only allows a single writer per
	// TimeSeries, see: https://cloud.google.com/monitoring/api/v3/metrics-details#intro-time-series
	// A TimeSeries is uniquely defined by the metric type name
	// (constructed from the view name and the MetricPrefix), the MonitoredResource field,
	// and the set of label key/value pairs (in OpenCensus terminology: tag).
	//
	// If no custom MonitoredResource is set AND if Resource is also not set then
	// a default MonitoredResource with type global and no resource labels will be used.
	// If you explicitly set this field, you may also want to set custom DefaultMonitoringLabels.
	//
	// This field replaces Resource field. If this is set then it will override the
	// Resource field.
	// Optional, but encouraged.
	MonitoredResource monitoredresource.Interface

	// MetricPrefix overrides the prefix of a Stackdriver metric display names.
	// Optional. If unset defaults to "OpenCensus/".
	// Deprecated: Provide GetMetricDisplayName to change the display name of
	// the metric.
	// If GetMetricDisplayName is non-nil, this option is ignored.
	MetricPrefix string

	// GetMetricDisplayName allows customizing the display name for the metric
	// associated with the given view. By default it will be:
	//   MetricPrefix + view.Name
	GetMetricDisplayName func(view *view.View) string

	// GetMetricType allows customizing the metric type for the given view.
	// By default, it will be:
	//   "custom.googleapis.com/opencensus/" + view.Name
	//
	// See: https://cloud.google.com/monitoring/api/ref_v3/rest/v3/projects.metricDescriptors#MetricDescriptor
	GetMetricType func(view *view.View) string

	// DefaultTraceAttributes will be appended to every span that is exported to
	// Stackdriver Trace.
	DefaultTraceAttributes map[string]interface{}

	// DefaultMonitoringLabels are labels added to every metric created by this
	// exporter in Stackdriver Monitoring.
	//
	// If unset, this defaults to a single label with key "opencensus_task" and
	// value "go-<pid>@<hostname>". This default ensures that the set of labels
	// together with the default Resource (global) are unique to this
	// process, as required by Stackdriver Monitoring.
	//
	// If you set DefaultMonitoringLabels, make sure that the Resource field
	// together with these labels is unique to the
	// current process. This is to ensure that there is only a single writer to
	// each TimeSeries in Stackdriver.
	//
	// Set this to &Labels{} (a pointer to an empty Labels) to avoid getting the
	// default "opencensus_task" label. You should only do this if you know that
	// the Resource you set uniquely identifies this Go process.
	DefaultMonitoringLabels *Labels

	// Context allows users to provide a custom context for API calls.
	//
	// This context will be used several times: first, to create Stackdriver
	// trace and metric clients, and then every time a new batch of traces or
	// stats needs to be uploaded.
	//
	// If unset, context.Background() will be used.
	Context context.Context
}

// Exporter is a stats.Exporter and trace.Exporter
// implementation that uploads data to Stackdriver.
type Exporter struct {
	traceExporter *traceExporter
	statsExporter *statsExporter
}

// NewExporter creates a new Exporter that implements both stats.Exporter and
// trace.Exporter.
func NewExporter(o Options) (*Exporter, error) {
	if o.Context == nil {
		o.Context = context.Background()
	}
	if o.ProjectID == "" {
		creds, err := google.FindDefaultCredentials(o.Context, traceapi.DefaultAuthScopes()...)
		if err != nil {
			return nil, fmt.Errorf("stackdriver: %v", err)
		}
		if creds.ProjectID == "" {
			return nil, errors.New("stackdriver: no project found with application default credentials")
		}
		o.ProjectID = creds.ProjectID
	}

	if o.MonitoredResource != nil {
		o.Resource = convertMonitoredResourceToPB(o.MonitoredResource)
	}

	se, err := newStatsExporter(o)
	if err != nil {
		return nil, err
	}
	te, err := newTraceExporter(o)
	if err != nil {
		return nil, err
	}
	return &Exporter{
		statsExporter: se,
		traceExporter: te,
	}, nil
}

// ExportView exports to the Stackdriver Monitoring if view data
// has one or more rows.
func (e *Exporter) ExportView(vd *view.Data) {
	e.statsExporter.ExportView(vd)
}

// ExportSpan exports a SpanData to Stackdriver Trace.
func (e *Exporter) ExportSpan(sd *trace.SpanData) {
	if len(e.traceExporter.o.DefaultTraceAttributes) > 0 {
		sd = e.sdWithDefaultTraceAttributes(sd)
	}
	e.traceExporter.ExportSpan(sd)
}

func (e *Exporter) sdWithDefaultTraceAttributes(sd *trace.SpanData) *trace.SpanData {
	newSD := *sd
	newSD.Attributes = make(map[string]interface{})
	for k, v := range e.traceExporter.o.DefaultTraceAttributes {
		newSD.Attributes[k] = v
	}
	for k, v := range sd.Attributes {
		newSD.Attributes[k] = v
	}
	return &newSD
}

// Flush waits for exported data to be uploaded.
//
// This is useful if your program is ending and you do not
// want to lose recent stats or spans.
func (e *Exporter) Flush() {
	e.statsExporter.Flush()
	e.traceExporter.Flush()
}

func (o Options) handleError(err error) {
	if o.OnError != nil {
		o.OnError(err)
		return
	}
	log.Printf("Failed to export to Stackdriver: %v", err)
}

// convertMonitoredResourceToPB converts MonitoredResource data in to
// protocol buffer.
func convertMonitoredResourceToPB(mr monitoredresource.Interface) *monitoredrespb.MonitoredResource {
	mrpb := new(monitoredrespb.MonitoredResource)
	var labels map[string]string
	mrpb.Type, labels = mr.MonitoredResource()
	mrpb.Labels = make(map[string]string)
	for k, v := range labels {
		mrpb.Labels[k] = v
	}
	return mrpb
}
