// Copyright 2017 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package v1 provides bindings to the Prometheus HTTP API v1:
// http://prometheus.io/docs/querying/api/
package v1

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unsafe"

	json "github.com/json-iterator/go"

	"github.com/prometheus/common/model"

	"github.com/prometheus/client_golang/api"
)

func init() {
	json.RegisterTypeEncoderFunc("model.SamplePair", marshalPointJSON, marshalPointJSONIsEmpty)
	json.RegisterTypeDecoderFunc("model.SamplePair", unMarshalPointJSON)
}

func unMarshalPointJSON(ptr unsafe.Pointer, iter *json.Iterator) {
	p := (*model.SamplePair)(ptr)
	if !iter.ReadArray() {
		iter.ReportError("unmarshal model.SamplePair", "SamplePair must be [timestamp, value]")
		return
	}
	t := iter.ReadNumber()
	if err := p.Timestamp.UnmarshalJSON([]byte(t)); err != nil {
		iter.ReportError("unmarshal model.SamplePair", err.Error())
		return
	}
	if !iter.ReadArray() {
		iter.ReportError("unmarshal model.SamplePair", "SamplePair missing value")
		return
	}

	f, err := strconv.ParseFloat(iter.ReadString(), 64)
	if err != nil {
		iter.ReportError("unmarshal model.SamplePair", err.Error())
		return
	}
	p.Value = model.SampleValue(f)

	if iter.ReadArray() {
		iter.ReportError("unmarshal model.SamplePair", "SamplePair has too many values, must be [timestamp, value]")
		return
	}
}

func marshalPointJSON(ptr unsafe.Pointer, stream *json.Stream) {
	p := *((*model.SamplePair)(ptr))
	stream.WriteArrayStart()
	// Write out the timestamp as a float divided by 1000.
	// This is ~3x faster than converting to a float.
	t := int64(p.Timestamp)
	if t < 0 {
		stream.WriteRaw(`-`)
		t = -t
	}
	stream.WriteInt64(t / 1000)
	fraction := t % 1000
	if fraction != 0 {
		stream.WriteRaw(`.`)
		if fraction < 100 {
			stream.WriteRaw(`0`)
		}
		if fraction < 10 {
			stream.WriteRaw(`0`)
		}
		stream.WriteInt64(fraction)
	}
	stream.WriteMore()
	stream.WriteRaw(`"`)

	// Taken from https://github.com/json-iterator/go/blob/master/stream_float.go#L71 as a workaround
	// to https://github.com/json-iterator/go/issues/365 (jsoniter, to follow json standard, doesn't allow inf/nan)
	buf := stream.Buffer()
	abs := math.Abs(float64(p.Value))
	fmt := byte('f')
	// Note: Must use float32 comparisons for underlying float32 value to get precise cutoffs right.
	if abs != 0 {
		if abs < 1e-6 || abs >= 1e21 {
			fmt = 'e'
		}
	}
	buf = strconv.AppendFloat(buf, float64(p.Value), fmt, -1, 64)
	stream.SetBuffer(buf)

	stream.WriteRaw(`"`)
	stream.WriteArrayEnd()

}

func marshalPointJSONIsEmpty(ptr unsafe.Pointer) bool {
	return false
}

const (
	statusAPIError = 422

	apiPrefix = "/api/v1"

	epAlerts          = apiPrefix + "/alerts"
	epAlertManagers   = apiPrefix + "/alertmanagers"
	epQuery           = apiPrefix + "/query"
	epQueryRange      = apiPrefix + "/query_range"
	epLabels          = apiPrefix + "/labels"
	epLabelValues     = apiPrefix + "/label/:name/values"
	epSeries          = apiPrefix + "/series"
	epTargets         = apiPrefix + "/targets"
	epTargetsMetadata = apiPrefix + "/targets/metadata"
	epMetadata        = apiPrefix + "/metadata"
	epRules           = apiPrefix + "/rules"
	epSnapshot        = apiPrefix + "/admin/tsdb/snapshot"
	epDeleteSeries    = apiPrefix + "/admin/tsdb/delete_series"
	epCleanTombstones = apiPrefix + "/admin/tsdb/clean_tombstones"
	epConfig          = apiPrefix + "/status/config"
	epFlags           = apiPrefix + "/status/flags"
	epRuntimeinfo     = apiPrefix + "/status/runtimeinfo"
)

// AlertState models the state of an alert.
type AlertState string

// ErrorType models the different API error types.
type ErrorType string

// HealthStatus models the health status of a scrape target.
type HealthStatus string

// RuleType models the type of a rule.
type RuleType string

// RuleHealth models the health status of a rule.
type RuleHealth string

// MetricType models the type of a metric.
type MetricType string

const (
	// Possible values for AlertState.
	AlertStateFiring   AlertState = "firing"
	AlertStateInactive AlertState = "inactive"
	AlertStatePending  AlertState = "pending"

	// Possible values for ErrorType.
	ErrBadData     ErrorType = "bad_data"
	ErrTimeout     ErrorType = "timeout"
	ErrCanceled    ErrorType = "canceled"
	ErrExec        ErrorType = "execution"
	ErrBadResponse ErrorType = "bad_response"
	ErrServer      ErrorType = "server_error"
	ErrClient      ErrorType = "client_error"

	// Possible values for HealthStatus.
	HealthGood    HealthStatus = "up"
	HealthUnknown HealthStatus = "unknown"
	HealthBad     HealthStatus = "down"

	// Possible values for RuleType.
	RuleTypeRecording RuleType = "recording"
	RuleTypeAlerting  RuleType = "alerting"

	// Possible values for RuleHealth.
	RuleHealthGood    = "ok"
	RuleHealthUnknown = "unknown"
	RuleHealthBad     = "err"

	// Possible values for MetricType
	MetricTypeCounter        MetricType = "counter"
	MetricTypeGauge          MetricType = "gauge"
	MetricTypeHistogram      MetricType = "histogram"
	MetricTypeGaugeHistogram MetricType = "gaugehistogram"
	MetricTypeSummary        MetricType = "summary"
	MetricTypeInfo           MetricType = "info"
	MetricTypeStateset       MetricType = "stateset"
	MetricTypeUnknown        MetricType = "unknown"
)

// Error is an error returned by the API.
type Error struct {
	Type   ErrorType
	Msg    string
	Detail string
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s: %s", e.Type, e.Msg)
}

// Range represents a sliced time range.
type Range struct {
	// The boundaries of the time range.
	Start, End time.Time
	// The maximum time between two slices within the boundaries.
	Step time.Duration
}

// API provides bindings for Prometheus's v1 API.
type API interface {
	// Alerts returns a list of all active alerts.
	Alerts(ctx context.Context) (AlertsResult, error)
	// AlertManagers returns an overview of the current state of the Prometheus alert manager discovery.
	AlertManagers(ctx context.Context) (AlertManagersResult, error)
	// CleanTombstones removes the deleted data from disk and cleans up the existing tombstones.
	CleanTombstones(ctx context.Context) error
	// Config returns the current Prometheus configuration.
	Config(ctx context.Context) (ConfigResult, error)
	// DeleteSeries deletes data for a selection of series in a time range.
	DeleteSeries(ctx context.Context, matches []string, startTime time.Time, endTime time.Time) error
	// Flags returns the flag values that Prometheus was launched with.
	Flags(ctx context.Context) (FlagsResult, error)
	// LabelNames returns all the unique label names present in the block in sorted order.
	LabelNames(ctx context.Context, startTime time.Time, endTime time.Time) ([]string, Warnings, error)
	// LabelValues performs a query for the values of the given label.
	LabelValues(ctx context.Context, label string, startTime time.Time, endTime time.Time) (model.LabelValues, Warnings, error)
	// Query performs a query for the given time.
	Query(ctx context.Context, query string, ts time.Time) (model.Value, Warnings, error)
	// QueryRange performs a query for the given range.
	QueryRange(ctx context.Context, query string, r Range) (model.Value, Warnings, error)
	// Runtimeinfo returns the various runtime information properties about the Prometheus server.
	Runtimeinfo(ctx context.Context) (RuntimeinfoResult, error)
	// Series finds series by label matchers.
	Series(ctx context.Context, matches []string, startTime time.Time, endTime time.Time) ([]model.LabelSet, Warnings, error)
	// Snapshot creates a snapshot of all current data into snapshots/<datetime>-<rand>
	// under the TSDB's data directory and returns the directory as response.
	Snapshot(ctx context.Context, skipHead bool) (SnapshotResult, error)
	// Rules returns a list of alerting and recording rules that are currently loaded.
	Rules(ctx context.Context) (RulesResult, error)
	// Targets returns an overview of the current state of the Prometheus target discovery.
	Targets(ctx context.Context) (TargetsResult, error)
	// TargetsMetadata returns metadata about metrics currently scraped by the target.
	TargetsMetadata(ctx context.Context, matchTarget string, metric string, limit string) ([]MetricMetadata, error)
	// Metadata returns metadata about metrics currently scraped by the metric name.
	Metadata(ctx context.Context, metric string, limit string) (map[string][]Metadata, error)
}

// AlertsResult contains the result from querying the alerts endpoint.
type AlertsResult struct {
	Alerts []Alert `json:"alerts"`
}

// AlertManagersResult contains the result from querying the alertmanagers endpoint.
type AlertManagersResult struct {
	Active  []AlertManager `json:"activeAlertManagers"`
	Dropped []AlertManager `json:"droppedAlertManagers"`
}

// AlertManager models a configured Alert Manager.
type AlertManager struct {
	URL string `json:"url"`
}

// ConfigResult contains the result from querying the config endpoint.
type ConfigResult struct {
	YAML string `json:"yaml"`
}

// FlagsResult contains the result from querying the flag endpoint.
type FlagsResult map[string]string

// RuntimeinfoResult contains the result from querying the runtimeinfo endpoint.
type RuntimeinfoResult struct {
	StartTime           string `json:"startTime"`
	CWD                 string `json:"CWD"`
	ReloadConfigSuccess bool   `json:"reloadConfigSuccess"`
	LastConfigTime      string `json:"lastConfigTime"`
	ChunkCount          int    `json:"chunkCount"`
	TimeSeriesCount     int    `json:"timeSeriesCount"`
	CorruptionCount     int    `json:"corruptionCount"`
	GoroutineCount      int    `json:"goroutineCount"`
	GOMAXPROCS          int    `json:"GOMAXPROCS"`
	GOGC                string `json:"GOGC"`
	GODEBUG             string `json:"GODEBUG"`
	StorageRetention    string `json:"storageRetention"`
}

// SnapshotResult contains the result from querying the snapshot endpoint.
type SnapshotResult struct {
	Name string `json:"name"`
}

// RulesResult contains the result from querying the rules endpoint.
type RulesResult struct {
	Groups []RuleGroup `json:"groups"`
}

// RuleGroup models a rule group that contains a set of recording and alerting rules.
type RuleGroup struct {
	Name     string  `json:"name"`
	File     string  `json:"file"`
	Interval float64 `json:"interval"`
	Rules    Rules   `json:"rules"`
}

// Recording and alerting rules are stored in the same slice to preserve the order
// that rules are returned in by the API.
//
// Rule types can be determined using a type switch:
//   switch v := rule.(type) {
//   case RecordingRule:
//   	fmt.Print("got a recording rule")
//   case AlertingRule:
//   	fmt.Print("got a alerting rule")
//   default:
//   	fmt.Printf("unknown rule type %s", v)
//   }
type Rules []interface{}

// AlertingRule models a alerting rule.
type AlertingRule struct {
	Name        string         `json:"name"`
	Query       string         `json:"query"`
	Duration    float64        `json:"duration"`
	Labels      model.LabelSet `json:"labels"`
	Annotations model.LabelSet `json:"annotations"`
	Alerts      []*Alert       `json:"alerts"`
	Health      RuleHealth     `json:"health"`
	LastError   string         `json:"lastError,omitempty"`
}

// RecordingRule models a recording rule.
type RecordingRule struct {
	Name      string         `json:"name"`
	Query     string         `json:"query"`
	Labels    model.LabelSet `json:"labels,omitempty"`
	Health    RuleHealth     `json:"health"`
	LastError string         `json:"lastError,omitempty"`
}

// Alert models an active alert.
type Alert struct {
	ActiveAt    time.Time `json:"activeAt"`
	Annotations model.LabelSet
	Labels      model.LabelSet
	State       AlertState
	Value       string
}

// TargetsResult contains the result from querying the targets endpoint.
type TargetsResult struct {
	Active  []ActiveTarget  `json:"activeTargets"`
	Dropped []DroppedTarget `json:"droppedTargets"`
}

// ActiveTarget models an active Prometheus scrape target.
type ActiveTarget struct {
	DiscoveredLabels map[string]string `json:"discoveredLabels"`
	Labels           model.LabelSet    `json:"labels"`
	ScrapeURL        string            `json:"scrapeUrl"`
	LastError        string            `json:"lastError"`
	LastScrape       time.Time         `json:"lastScrape"`
	Health           HealthStatus      `json:"health"`
}

// DroppedTarget models a dropped Prometheus scrape target.
type DroppedTarget struct {
	DiscoveredLabels map[string]string `json:"discoveredLabels"`
}

// MetricMetadata models the metadata of a metric with its scrape target and name.
type MetricMetadata struct {
	Target map[string]string `json:"target"`
	Metric string            `json:"metric,omitempty"`
	Type   MetricType        `json:"type"`
	Help   string            `json:"help"`
	Unit   string            `json:"unit"`
}

// Metadata models the metadata of a metric.
type Metadata struct {
	Type MetricType `json:"type"`
	Help string     `json:"help"`
	Unit string     `json:"unit"`
}

// queryResult contains result data for a query.
type queryResult struct {
	Type   model.ValueType `json:"resultType"`
	Result interface{}     `json:"result"`

	// The decoded value.
	v model.Value
}

func (rg *RuleGroup) UnmarshalJSON(b []byte) error {
	v := struct {
		Name     string            `json:"name"`
		File     string            `json:"file"`
		Interval float64           `json:"interval"`
		Rules    []json.RawMessage `json:"rules"`
	}{}

	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}

	rg.Name = v.Name
	rg.File = v.File
	rg.Interval = v.Interval

	for _, rule := range v.Rules {
		alertingRule := AlertingRule{}
		if err := json.Unmarshal(rule, &alertingRule); err == nil {
			rg.Rules = append(rg.Rules, alertingRule)
			continue
		}
		recordingRule := RecordingRule{}
		if err := json.Unmarshal(rule, &recordingRule); err == nil {
			rg.Rules = append(rg.Rules, recordingRule)
			continue
		}
		return errors.New("failed to decode JSON into an alerting or recording rule")
	}

	return nil
}

func (r *AlertingRule) UnmarshalJSON(b []byte) error {
	v := struct {
		Type string `json:"type"`
	}{}
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	if v.Type == "" {
		return errors.New("type field not present in rule")
	}
	if v.Type != string(RuleTypeAlerting) {
		return fmt.Errorf("expected rule of type %s but got %s", string(RuleTypeAlerting), v.Type)
	}

	rule := struct {
		Name        string         `json:"name"`
		Query       string         `json:"query"`
		Duration    float64        `json:"duration"`
		Labels      model.LabelSet `json:"labels"`
		Annotations model.LabelSet `json:"annotations"`
		Alerts      []*Alert       `json:"alerts"`
		Health      RuleHealth     `json:"health"`
		LastError   string         `json:"lastError,omitempty"`
	}{}
	if err := json.Unmarshal(b, &rule); err != nil {
		return err
	}
	r.Health = rule.Health
	r.Annotations = rule.Annotations
	r.Name = rule.Name
	r.Query = rule.Query
	r.Alerts = rule.Alerts
	r.Duration = rule.Duration
	r.Labels = rule.Labels
	r.LastError = rule.LastError

	return nil
}

func (r *RecordingRule) UnmarshalJSON(b []byte) error {
	v := struct {
		Type string `json:"type"`
	}{}
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	if v.Type == "" {
		return errors.New("type field not present in rule")
	}
	if v.Type != string(RuleTypeRecording) {
		return fmt.Errorf("expected rule of type %s but got %s", string(RuleTypeRecording), v.Type)
	}

	rule := struct {
		Name      string         `json:"name"`
		Query     string         `json:"query"`
		Labels    model.LabelSet `json:"labels,omitempty"`
		Health    RuleHealth     `json:"health"`
		LastError string         `json:"lastError,omitempty"`
	}{}
	if err := json.Unmarshal(b, &rule); err != nil {
		return err
	}
	r.Health = rule.Health
	r.Labels = rule.Labels
	r.Name = rule.Name
	r.LastError = rule.LastError
	r.Query = rule.Query

	return nil
}

func (qr *queryResult) UnmarshalJSON(b []byte) error {
	v := struct {
		Type   model.ValueType `json:"resultType"`
		Result json.RawMessage `json:"result"`
	}{}

	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}

	switch v.Type {
	case model.ValScalar:
		var sv model.Scalar
		err = json.Unmarshal(v.Result, &sv)
		qr.v = &sv

	case model.ValVector:
		var vv model.Vector
		err = json.Unmarshal(v.Result, &vv)
		qr.v = vv

	case model.ValMatrix:
		var mv model.Matrix
		err = json.Unmarshal(v.Result, &mv)
		qr.v = mv

	default:
		err = fmt.Errorf("unexpected value type %q", v.Type)
	}
	return err
}

// NewAPI returns a new API for the client.
//
// It is safe to use the returned API from multiple goroutines.
func NewAPI(c api.Client) API {
	return &httpAPI{
		client: &apiClientImpl{
			client: c,
		},
	}
}

type httpAPI struct {
	client apiClient
}

func (h *httpAPI) Alerts(ctx context.Context) (AlertsResult, error) {
	u := h.client.URL(epAlerts, nil)

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return AlertsResult{}, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return AlertsResult{}, err
	}

	var res AlertsResult
	return res, json.Unmarshal(body, &res)
}

func (h *httpAPI) AlertManagers(ctx context.Context) (AlertManagersResult, error) {
	u := h.client.URL(epAlertManagers, nil)

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return AlertManagersResult{}, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return AlertManagersResult{}, err
	}

	var res AlertManagersResult
	return res, json.Unmarshal(body, &res)
}

func (h *httpAPI) CleanTombstones(ctx context.Context) error {
	u := h.client.URL(epCleanTombstones, nil)

	req, err := http.NewRequest(http.MethodPost, u.String(), nil)
	if err != nil {
		return err
	}

	_, _, _, err = h.client.Do(ctx, req)
	return err
}

func (h *httpAPI) Config(ctx context.Context) (ConfigResult, error) {
	u := h.client.URL(epConfig, nil)

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return ConfigResult{}, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return ConfigResult{}, err
	}

	var res ConfigResult
	return res, json.Unmarshal(body, &res)
}

func (h *httpAPI) DeleteSeries(ctx context.Context, matches []string, startTime time.Time, endTime time.Time) error {
	u := h.client.URL(epDeleteSeries, nil)
	q := u.Query()

	for _, m := range matches {
		q.Add("match[]", m)
	}

	q.Set("start", formatTime(startTime))
	q.Set("end", formatTime(endTime))

	u.RawQuery = q.Encode()

	req, err := http.NewRequest(http.MethodPost, u.String(), nil)
	if err != nil {
		return err
	}

	_, _, _, err = h.client.Do(ctx, req)
	return err
}

func (h *httpAPI) Flags(ctx context.Context) (FlagsResult, error) {
	u := h.client.URL(epFlags, nil)

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return FlagsResult{}, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return FlagsResult{}, err
	}

	var res FlagsResult
	return res, json.Unmarshal(body, &res)
}

func (h *httpAPI) Runtimeinfo(ctx context.Context) (RuntimeinfoResult, error) {
	u := h.client.URL(epRuntimeinfo, nil)

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return RuntimeinfoResult{}, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return RuntimeinfoResult{}, err
	}

	var res RuntimeinfoResult
	return res, json.Unmarshal(body, &res)
}

func (h *httpAPI) LabelNames(ctx context.Context, startTime time.Time, endTime time.Time) ([]string, Warnings, error) {
	u := h.client.URL(epLabels, nil)
	q := u.Query()
	q.Set("start", formatTime(startTime))
	q.Set("end", formatTime(endTime))

	u.RawQuery = q.Encode()

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, nil, err
	}
	_, body, w, err := h.client.Do(ctx, req)
	if err != nil {
		return nil, w, err
	}
	var labelNames []string
	return labelNames, w, json.Unmarshal(body, &labelNames)
}

func (h *httpAPI) LabelValues(ctx context.Context, label string, startTime time.Time, endTime time.Time) (model.LabelValues, Warnings, error) {
	u := h.client.URL(epLabelValues, map[string]string{"name": label})
	q := u.Query()
	q.Set("start", formatTime(startTime))
	q.Set("end", formatTime(endTime))

	u.RawQuery = q.Encode()

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, nil, err
	}
	_, body, w, err := h.client.Do(ctx, req)
	if err != nil {
		return nil, w, err
	}
	var labelValues model.LabelValues
	return labelValues, w, json.Unmarshal(body, &labelValues)
}

func (h *httpAPI) Query(ctx context.Context, query string, ts time.Time) (model.Value, Warnings, error) {
	u := h.client.URL(epQuery, nil)
	q := u.Query()

	q.Set("query", query)
	if !ts.IsZero() {
		q.Set("time", formatTime(ts))
	}

	_, body, warnings, err := h.client.DoGetFallback(ctx, u, q)
	if err != nil {
		return nil, warnings, err
	}

	var qres queryResult
	return model.Value(qres.v), warnings, json.Unmarshal(body, &qres)
}

func (h *httpAPI) QueryRange(ctx context.Context, query string, r Range) (model.Value, Warnings, error) {
	u := h.client.URL(epQueryRange, nil)
	q := u.Query()

	q.Set("query", query)
	q.Set("start", formatTime(r.Start))
	q.Set("end", formatTime(r.End))
	q.Set("step", strconv.FormatFloat(r.Step.Seconds(), 'f', -1, 64))

	_, body, warnings, err := h.client.DoGetFallback(ctx, u, q)
	if err != nil {
		return nil, warnings, err
	}

	var qres queryResult

	return model.Value(qres.v), warnings, json.Unmarshal(body, &qres)
}

func (h *httpAPI) Series(ctx context.Context, matches []string, startTime time.Time, endTime time.Time) ([]model.LabelSet, Warnings, error) {
	u := h.client.URL(epSeries, nil)
	q := u.Query()

	for _, m := range matches {
		q.Add("match[]", m)
	}

	q.Set("start", formatTime(startTime))
	q.Set("end", formatTime(endTime))

	u.RawQuery = q.Encode()

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, nil, err
	}

	_, body, warnings, err := h.client.Do(ctx, req)
	if err != nil {
		return nil, warnings, err
	}

	var mset []model.LabelSet
	return mset, warnings, json.Unmarshal(body, &mset)
}

func (h *httpAPI) Snapshot(ctx context.Context, skipHead bool) (SnapshotResult, error) {
	u := h.client.URL(epSnapshot, nil)
	q := u.Query()

	q.Set("skip_head", strconv.FormatBool(skipHead))

	u.RawQuery = q.Encode()

	req, err := http.NewRequest(http.MethodPost, u.String(), nil)
	if err != nil {
		return SnapshotResult{}, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return SnapshotResult{}, err
	}

	var res SnapshotResult
	return res, json.Unmarshal(body, &res)
}

func (h *httpAPI) Rules(ctx context.Context) (RulesResult, error) {
	u := h.client.URL(epRules, nil)

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return RulesResult{}, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return RulesResult{}, err
	}

	var res RulesResult
	return res, json.Unmarshal(body, &res)
}

func (h *httpAPI) Targets(ctx context.Context) (TargetsResult, error) {
	u := h.client.URL(epTargets, nil)

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return TargetsResult{}, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return TargetsResult{}, err
	}

	var res TargetsResult
	return res, json.Unmarshal(body, &res)
}

func (h *httpAPI) TargetsMetadata(ctx context.Context, matchTarget string, metric string, limit string) ([]MetricMetadata, error) {
	u := h.client.URL(epTargetsMetadata, nil)
	q := u.Query()

	q.Set("match_target", matchTarget)
	q.Set("metric", metric)
	q.Set("limit", limit)

	u.RawQuery = q.Encode()

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return nil, err
	}

	var res []MetricMetadata
	return res, json.Unmarshal(body, &res)
}

func (h *httpAPI) Metadata(ctx context.Context, metric string, limit string) (map[string][]Metadata, error) {
	u := h.client.URL(epMetadata, nil)
	q := u.Query()

	q.Set("metric", metric)
	q.Set("limit", limit)

	u.RawQuery = q.Encode()

	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}

	_, body, _, err := h.client.Do(ctx, req)
	if err != nil {
		return nil, err
	}

	var res map[string][]Metadata
	return res, json.Unmarshal(body, &res)
}

// Warnings is an array of non critical errors
type Warnings []string

// apiClient wraps a regular client and processes successful API responses.
// Successful also includes responses that errored at the API level.
type apiClient interface {
	URL(ep string, args map[string]string) *url.URL
	Do(context.Context, *http.Request) (*http.Response, []byte, Warnings, error)
	DoGetFallback(ctx context.Context, u *url.URL, args url.Values) (*http.Response, []byte, Warnings, error)
}

type apiClientImpl struct {
	client api.Client
}

type apiResponse struct {
	Status    string          `json:"status"`
	Data      json.RawMessage `json:"data"`
	ErrorType ErrorType       `json:"errorType"`
	Error     string          `json:"error"`
	Warnings  []string        `json:"warnings,omitempty"`
}

func apiError(code int) bool {
	// These are the codes that Prometheus sends when it returns an error.
	return code == statusAPIError || code == http.StatusBadRequest
}

func errorTypeAndMsgFor(resp *http.Response) (ErrorType, string) {
	switch resp.StatusCode / 100 {
	case 4:
		return ErrClient, fmt.Sprintf("client error: %d", resp.StatusCode)
	case 5:
		return ErrServer, fmt.Sprintf("server error: %d", resp.StatusCode)
	}
	return ErrBadResponse, fmt.Sprintf("bad response code %d", resp.StatusCode)
}

func (h *apiClientImpl) URL(ep string, args map[string]string) *url.URL {
	return h.client.URL(ep, args)
}

func (h *apiClientImpl) Do(ctx context.Context, req *http.Request) (*http.Response, []byte, Warnings, error) {
	resp, body, err := h.client.Do(ctx, req)
	if err != nil {
		return resp, body, nil, err
	}

	code := resp.StatusCode

	if code/100 != 2 && !apiError(code) {
		errorType, errorMsg := errorTypeAndMsgFor(resp)
		return resp, body, nil, &Error{
			Type:   errorType,
			Msg:    errorMsg,
			Detail: string(body),
		}
	}

	var result apiResponse

	if http.StatusNoContent != code {
		if jsonErr := json.Unmarshal(body, &result); jsonErr != nil {
			return resp, body, nil, &Error{
				Type: ErrBadResponse,
				Msg:  jsonErr.Error(),
			}
		}
	}

	if apiError(code) && result.Status == "success" {
		err = &Error{
			Type: ErrBadResponse,
			Msg:  "inconsistent body for response code",
		}
	}

	if result.Status == "error" {
		err = &Error{
			Type: result.ErrorType,
			Msg:  result.Error,
		}
	}

	return resp, []byte(result.Data), result.Warnings, err

}

// DoGetFallback will attempt to do the request as-is, and on a 405 it will fallback to a GET request.
func (h *apiClientImpl) DoGetFallback(ctx context.Context, u *url.URL, args url.Values) (*http.Response, []byte, Warnings, error) {
	req, err := http.NewRequest(http.MethodPost, u.String(), strings.NewReader(args.Encode()))
	if err != nil {
		return nil, nil, nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, body, warnings, err := h.Do(ctx, req)
	if resp != nil && resp.StatusCode == http.StatusMethodNotAllowed {
		u.RawQuery = args.Encode()
		req, err = http.NewRequest(http.MethodGet, u.String(), nil)
		if err != nil {
			return nil, nil, warnings, err
		}

	} else {
		if err != nil {
			return resp, body, warnings, err
		}
		return resp, body, warnings, nil
	}
	return h.Do(ctx, req)
}

func formatTime(t time.Time) string {
	return strconv.FormatFloat(float64(t.Unix())+float64(t.Nanosecond())/1e9, 'f', -1, 64)
}
