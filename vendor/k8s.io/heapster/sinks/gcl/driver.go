// Copyright 2015 Google Inc. All Rights Reserved.
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

package gcl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"time"

	"github.com/golang/glog"
	"google.golang.org/cloud/compute/metadata"
	"k8s.io/heapster/extpoints"
	"k8s.io/heapster/util/gce"

	sink_api "k8s.io/heapster/sinks/api"
	kube_api "k8s.io/kubernetes/pkg/api"
)

const (
	GCLAuthScope             = "https://www.googleapis.com/auth/logging.write"
	eventLoggingSeverity     = "NOTICE"
	eventLoggingServiceName  = "custom.googleapis.com"
	eventLoggingLogName      = "kubernetes.io/events"
	logEntriesWriteURLScheme = "https"
	logEntriesWriteURLHost   = "logging.googleapis.com"
	logEntriesWriteURLFormat = "/v1beta3/projects/%s/logs/%s/entries:write"
)

func (sink *gclSink) DebugInfo() string {
	return fmt.Sprintf("Sink Type: Google Cloud Logging (GCL). Http Error Count: %v\r\n", sink.httpErrorCount)
}

func (sink *gclSink) Register(metrics []sink_api.MetricDescriptor) error {
	// No-op
	return nil
}

func (sink *gclSink) Unregister(metrics []sink_api.MetricDescriptor) error {
	// No-op
	return nil
}

// Stores metrics into the backend
func (sink *gclSink) StoreTimeseries(input []sink_api.Timeseries) error {
	// No-op, Google Cloud Logging (GCL) doesn't store metrics
	return nil
}

// Stores events into the backend.
func (sink *gclSink) StoreEvents(events []kube_api.Event) error {
	if events == nil || len(events) <= 0 {
		return nil
	}
	glog.V(3).Infof("storing events into GCL sink")
	request := sink.createLogsEntriesRequest(events)
	err := sink.sendLogsEntriesRequest(request)
	glog.V(3).Infof("stored events into GCL sink - %v", err)
	return err
}

type httpClient interface {
	Do(req *http.Request) (*http.Response, error)
}

type gclSink struct {
	// Token to use for authentication.
	token gce.AuthTokenProvider

	// GCE Project ID
	projectId string

	// HTTP Client
	httpClient httpClient

	// Number of times an Http Error was encountered (for debugging)
	httpErrorCount uint
}

type LogsEntriesWriteRequest struct {
	// Metadata labels that apply to all log entries in this request, so they don't have to be
	// repeated in each log entry's metadata.labels field. If any of the log entries contain
	// a (key, value) with the same key that is in commonLabels, then the entry's (key, value)
	// overrides the one in commonLabels.
	CommonLabels map[string]string `json:"commonLabels,omitempty"`

	// Entries: Log entries to insert.
	Entries []*LogEntry `json:"entries,omitempty"`
}

type LogEntry struct {
	// Information about the log entry.
	Metadata *LogEntryMetadata `json:"metadata,omitempty"`

	// A unique ID for the log entry. If this field is provided and is identical to a previously
	// created entry, then the previous instance of this entry is replaced with this one.
	InsertId string `json:"insertId,omitempty"`

	// The log to which this entry belongs. When a log entry is ingested, the value of this field
	// is set by the logging system.
	Log string `json:"log,omitempty"`

	// The log entry payload, represented by "JSON-like" structured data, in our case, the event.
	Payload kube_api.Event `json:"structPayload,omitempty"`
}

type LogEntryMetadata struct {
	// The time the event described by the log entry occurred. Timestamps must be later than
	// January 1, 1970. Timestamp must be a string following RFC 3339, it must be Z-normalized and
	// use 3, 6, or 9 fractional digits depending on required precision.
	Timestamp string `json:"timestamp,omitempty"`

	// The severity of the log entry. Acceptable values:
	// DEFAULT - The log entry has no assigned severity level.
	// DEBUG - Debug or trace information.
	// INFO - Routine information, such as ongoing status or performance.
	// NOTICE - Normal but significant events, such as start up, shut down, or configuration.
	// WARNING - Warning events might cause problems.
	// ERROR - Error events are likely to cause problems.
	// CRITICAL - Critical events cause more severe problems or brief outages.
	// ALERT - A person must take an action immediately.
	// EMERGENCY - One or more systems are unusable.
	Severity string `json:"severity,omitempty"`

	// The project ID of the Google Cloud Platform service that created the log entry.
	ProjectId string `json:"projectId,omitempty"`

	// The API name of the Google Cloud Platform service that created the log entry.
	// For example, "compute.googleapis.com" or "custom.googleapis.com".
	ServiceName string `json:"serviceName,omitempty"`

	// The region name of the Google Cloud Platform service that created the log entry.
	// For example, `"us-central1"`.
	Region string `json:"region,omitempty"`

	// The zone of the Google Cloud Platform service that created the log entry.
	// For example, `"us-central1-a"`.
	Zone string `json:"zone,omitempty"`

	// The fully-qualified email address of the authenticated user that performed or requested
	// the action represented by the log entry. If the log entry does not apply to an action
	// taken by an authenticated user, then the field should be empty.
	UserId string `json:"userId,omitempty"`

	// A set of (key, value) data that provides additional information about the log entry.
	Labels map[string]string `json:"labels,omitempty"`
}

func (sink *gclSink) createLogsEntriesRequest(events []kube_api.Event) LogsEntriesWriteRequest {
	logEntries := make([]*LogEntry, len(events))
	for i, event := range events {
		logEntries[i] = &LogEntry{
			Metadata: &LogEntryMetadata{
				Timestamp:   event.LastTimestamp.Time.UTC().Format(time.RFC3339),
				Severity:    eventLoggingSeverity,
				ProjectId:   sink.projectId,
				ServiceName: eventLoggingServiceName,
			},
			InsertId: string(event.UID),
			Payload:  event,
		}
	}
	return LogsEntriesWriteRequest{Entries: logEntries}
}

// TODO: Move this to a common lib and share it with GCM implementation.
func (sink *gclSink) sendLogsEntriesRequest(request LogsEntriesWriteRequest) error {
	token, err := sink.token.GetToken()
	if err != nil {
		return err
	}

	requestBody, err := json.Marshal(request)
	if err != nil {
		return err
	}

	url := &url.URL{
		Scheme: logEntriesWriteURLScheme,
		Host:   logEntriesWriteURLHost,
		Opaque: fmt.Sprintf(logEntriesWriteURLFormat, sink.projectId, url.QueryEscape(eventLoggingLogName)),
	}

	req, err := http.NewRequest("POST", url.String(), bytes.NewReader(requestBody))
	if err != nil {
		return err
	}
	req.URL = url
	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("Authorization", fmt.Sprintf("Bearer %s", token))

	resp, err := sink.httpClient.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()
	out, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if resp.StatusCode != http.StatusOK {
		sink.httpErrorCount++
		return fmt.Errorf("request to %q failed with status %q and response: %q", url, resp.Status, string(out))
	}

	return nil
}

func (self *gclSink) Name() string {
	return "Google Cloud Logging Sink"
}

// Returns an implementation of a Google Cloud Logging (GCL) sink.
func new() (sink_api.ExternalSink, error) {
	token, err := gce.NewAuthTokenProvider(GCLAuthScope)
	if err != nil {
		return nil, err
	}

	// Detect project ID
	projectId, err := metadata.ProjectID()
	if err != nil {
		return nil, err
	}
	glog.Infof("Project ID for GCL sink is: %q\r\n", projectId)

	impl := &gclSink{
		token:      token,
		projectId:  projectId,
		httpClient: &http.Client{},
	}

	return impl, nil
}

func init() {
	extpoints.SinkFactories.Register(CreateGCLSink, "gcl")
}

func CreateGCLSink(uri *url.URL, _ extpoints.HeapsterConf) ([]sink_api.ExternalSink, error) {
	if *uri != (url.URL{}) {
		return nil, fmt.Errorf("gcl sinks don't take arguments")
	}
	sink, err := new()
	glog.Infof("creating GCL sink")
	return []sink_api.ExternalSink{sink}, err
}
