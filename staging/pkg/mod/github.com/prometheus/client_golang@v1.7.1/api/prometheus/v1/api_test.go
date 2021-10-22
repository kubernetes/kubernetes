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

package v1

import (
	"context"
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"time"

	json "github.com/json-iterator/go"

	"github.com/prometheus/common/model"
)

type apiTest struct {
	do           func() (interface{}, Warnings, error)
	inWarnings   []string
	inErr        error
	inStatusCode int
	inRes        interface{}

	reqPath   string
	reqParam  url.Values
	reqMethod string
	res       interface{}
	warnings  Warnings
	err       error
}

type apiTestClient struct {
	*testing.T
	curTest apiTest
}

func (c *apiTestClient) URL(ep string, args map[string]string) *url.URL {
	path := ep
	for k, v := range args {
		path = strings.Replace(path, ":"+k, v, -1)
	}
	u := &url.URL{
		Host: "test:9090",
		Path: path,
	}
	return u
}

func (c *apiTestClient) Do(ctx context.Context, req *http.Request) (*http.Response, []byte, Warnings, error) {

	test := c.curTest

	if req.URL.Path != test.reqPath {
		c.Errorf("unexpected request path: want %s, got %s", test.reqPath, req.URL.Path)
	}
	if req.Method != test.reqMethod {
		c.Errorf("unexpected request method: want %s, got %s", test.reqMethod, req.Method)
	}

	b, err := json.Marshal(test.inRes)
	if err != nil {
		c.Fatal(err)
	}

	resp := &http.Response{}
	if test.inStatusCode != 0 {
		resp.StatusCode = test.inStatusCode
	} else if test.inErr != nil {
		resp.StatusCode = statusAPIError
	} else {
		resp.StatusCode = http.StatusOK
	}

	return resp, b, test.inWarnings, test.inErr
}

func (c *apiTestClient) DoGetFallback(ctx context.Context, u *url.URL, args url.Values) (*http.Response, []byte, Warnings, error) {
	req, err := http.NewRequest(http.MethodPost, u.String(), strings.NewReader(args.Encode()))
	if err != nil {
		return nil, nil, nil, err
	}
	return c.Do(ctx, req)
}

func TestAPIs(t *testing.T) {

	testTime := time.Now()

	tc := &apiTestClient{
		T: t,
	}
	promAPI := &httpAPI{
		client: tc,
	}

	doAlertManagers := func() func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			v, err := promAPI.AlertManagers(context.Background())
			return v, nil, err
		}
	}

	doCleanTombstones := func() func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			return nil, nil, promAPI.CleanTombstones(context.Background())
		}
	}

	doConfig := func() func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			v, err := promAPI.Config(context.Background())
			return v, nil, err
		}
	}

	doDeleteSeries := func(matcher string, startTime time.Time, endTime time.Time) func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			return nil, nil, promAPI.DeleteSeries(context.Background(), []string{matcher}, startTime, endTime)
		}
	}

	doFlags := func() func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			v, err := promAPI.Flags(context.Background())
			return v, nil, err
		}
	}

	doRuntimeinfo := func() func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			v, err := promAPI.Runtimeinfo(context.Background())
			return v, nil, err
		}
	}

	doLabelNames := func(label string) func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			return promAPI.LabelNames(context.Background(), time.Now().Add(-100*time.Hour), time.Now())
		}
	}

	doLabelValues := func(label string) func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			return promAPI.LabelValues(context.Background(), label, time.Now().Add(-100*time.Hour), time.Now())
		}
	}

	doQuery := func(q string, ts time.Time) func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			return promAPI.Query(context.Background(), q, ts)
		}
	}

	doQueryRange := func(q string, rng Range) func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			return promAPI.QueryRange(context.Background(), q, rng)
		}
	}

	doSeries := func(matcher string, startTime time.Time, endTime time.Time) func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			return promAPI.Series(context.Background(), []string{matcher}, startTime, endTime)
		}
	}

	doSnapshot := func(skipHead bool) func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			v, err := promAPI.Snapshot(context.Background(), skipHead)
			return v, nil, err
		}
	}

	doRules := func() func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			v, err := promAPI.Rules(context.Background())
			return v, nil, err
		}
	}

	doTargets := func() func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			v, err := promAPI.Targets(context.Background())
			return v, nil, err
		}
	}

	doTargetsMetadata := func(matchTarget string, metric string, limit string) func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			v, err := promAPI.TargetsMetadata(context.Background(), matchTarget, metric, limit)
			return v, nil, err
		}
	}

	doMetadata := func(metric string, limit string) func() (interface{}, Warnings, error) {
		return func() (interface{}, Warnings, error) {
			v, err := promAPI.Metadata(context.Background(), metric, limit)
			return v, nil, err
		}
	}

	queryTests := []apiTest{
		{
			do: doQuery("2", testTime),
			inRes: &queryResult{
				Type: model.ValScalar,
				Result: &model.Scalar{
					Value:     2,
					Timestamp: model.TimeFromUnix(testTime.Unix()),
				},
			},

			reqMethod: "POST",
			reqPath:   "/api/v1/query",
			reqParam: url.Values{
				"query": []string{"2"},
				"time":  []string{testTime.Format(time.RFC3339Nano)},
			},
			res: &model.Scalar{
				Value:     2,
				Timestamp: model.TimeFromUnix(testTime.Unix()),
			},
		},
		{
			do:    doQuery("2", testTime),
			inErr: fmt.Errorf("some error"),

			reqMethod: "POST",
			reqPath:   "/api/v1/query",
			reqParam: url.Values{
				"query": []string{"2"},
				"time":  []string{testTime.Format(time.RFC3339Nano)},
			},
			err: fmt.Errorf("some error"),
		},
		{
			do:           doQuery("2", testTime),
			inRes:        "some body",
			inStatusCode: 500,
			inErr: &Error{
				Type:   ErrServer,
				Msg:    "server error: 500",
				Detail: "some body",
			},

			reqMethod: "POST",
			reqPath:   "/api/v1/query",
			reqParam: url.Values{
				"query": []string{"2"},
				"time":  []string{testTime.Format(time.RFC3339Nano)},
			},
			err: errors.New("server_error: server error: 500"),
		},
		{
			do:           doQuery("2", testTime),
			inRes:        "some body",
			inStatusCode: 404,
			inErr: &Error{
				Type:   ErrClient,
				Msg:    "client error: 404",
				Detail: "some body",
			},

			reqMethod: "POST",
			reqPath:   "/api/v1/query",
			reqParam: url.Values{
				"query": []string{"2"},
				"time":  []string{testTime.Format(time.RFC3339Nano)},
			},
			err: errors.New("client_error: client error: 404"),
		},
		// Warning only.
		{
			do:         doQuery("2", testTime),
			inWarnings: []string{"warning"},
			inRes: &queryResult{
				Type: model.ValScalar,
				Result: &model.Scalar{
					Value:     2,
					Timestamp: model.TimeFromUnix(testTime.Unix()),
				},
			},

			reqMethod: "POST",
			reqPath:   "/api/v1/query",
			reqParam: url.Values{
				"query": []string{"2"},
				"time":  []string{testTime.Format(time.RFC3339Nano)},
			},
			res: &model.Scalar{
				Value:     2,
				Timestamp: model.TimeFromUnix(testTime.Unix()),
			},
			warnings: []string{"warning"},
		},
		// Warning + error.
		{
			do:           doQuery("2", testTime),
			inWarnings:   []string{"warning"},
			inRes:        "some body",
			inStatusCode: 404,
			inErr: &Error{
				Type:   ErrClient,
				Msg:    "client error: 404",
				Detail: "some body",
			},

			reqMethod: "POST",
			reqPath:   "/api/v1/query",
			reqParam: url.Values{
				"query": []string{"2"},
				"time":  []string{testTime.Format(time.RFC3339Nano)},
			},
			err:      errors.New("client_error: client error: 404"),
			warnings: []string{"warning"},
		},

		{
			do: doQueryRange("2", Range{
				Start: testTime.Add(-time.Minute),
				End:   testTime,
				Step:  time.Minute,
			}),
			inErr: fmt.Errorf("some error"),

			reqMethod: "POST",
			reqPath:   "/api/v1/query_range",
			reqParam: url.Values{
				"query": []string{"2"},
				"start": []string{testTime.Add(-time.Minute).Format(time.RFC3339Nano)},
				"end":   []string{testTime.Format(time.RFC3339Nano)},
				"step":  []string{time.Minute.String()},
			},
			err: fmt.Errorf("some error"),
		},

		{
			do:        doLabelNames("mylabel"),
			inRes:     []string{"val1", "val2"},
			reqMethod: "GET",
			reqPath:   "/api/v1/labels",
			res:       []string{"val1", "val2"},
		},
		{
			do:         doLabelNames("mylabel"),
			inRes:      []string{"val1", "val2"},
			inWarnings: []string{"a"},
			reqMethod:  "GET",
			reqPath:    "/api/v1/labels",
			res:        []string{"val1", "val2"},
			warnings:   []string{"a"},
		},

		{
			do:        doLabelNames("mylabel"),
			inErr:     fmt.Errorf("some error"),
			reqMethod: "GET",
			reqPath:   "/api/v1/labels",
			err:       fmt.Errorf("some error"),
		},
		{
			do:         doLabelNames("mylabel"),
			inErr:      fmt.Errorf("some error"),
			inWarnings: []string{"a"},
			reqMethod:  "GET",
			reqPath:    "/api/v1/labels",
			err:        fmt.Errorf("some error"),
			warnings:   []string{"a"},
		},

		{
			do:        doLabelValues("mylabel"),
			inRes:     []string{"val1", "val2"},
			reqMethod: "GET",
			reqPath:   "/api/v1/label/mylabel/values",
			res:       model.LabelValues{"val1", "val2"},
		},
		{
			do:         doLabelValues("mylabel"),
			inRes:      []string{"val1", "val2"},
			inWarnings: []string{"a"},
			reqMethod:  "GET",
			reqPath:    "/api/v1/label/mylabel/values",
			res:        model.LabelValues{"val1", "val2"},
			warnings:   []string{"a"},
		},

		{
			do:        doLabelValues("mylabel"),
			inErr:     fmt.Errorf("some error"),
			reqMethod: "GET",
			reqPath:   "/api/v1/label/mylabel/values",
			err:       fmt.Errorf("some error"),
		},
		{
			do:         doLabelValues("mylabel"),
			inErr:      fmt.Errorf("some error"),
			inWarnings: []string{"a"},
			reqMethod:  "GET",
			reqPath:    "/api/v1/label/mylabel/values",
			err:        fmt.Errorf("some error"),
			warnings:   []string{"a"},
		},

		{
			do: doSeries("up", testTime.Add(-time.Minute), testTime),
			inRes: []map[string]string{
				{
					"__name__": "up",
					"job":      "prometheus",
					"instance": "localhost:9090"},
			},
			reqMethod: "GET",
			reqPath:   "/api/v1/series",
			reqParam: url.Values{
				"match": []string{"up"},
				"start": []string{testTime.Add(-time.Minute).Format(time.RFC3339Nano)},
				"end":   []string{testTime.Format(time.RFC3339Nano)},
			},
			res: []model.LabelSet{
				{
					"__name__": "up",
					"job":      "prometheus",
					"instance": "localhost:9090",
				},
			},
		},
		// Series with data + warning.
		{
			do: doSeries("up", testTime.Add(-time.Minute), testTime),
			inRes: []map[string]string{
				{
					"__name__": "up",
					"job":      "prometheus",
					"instance": "localhost:9090"},
			},
			inWarnings: []string{"a"},
			reqMethod:  "GET",
			reqPath:    "/api/v1/series",
			reqParam: url.Values{
				"match": []string{"up"},
				"start": []string{testTime.Add(-time.Minute).Format(time.RFC3339Nano)},
				"end":   []string{testTime.Format(time.RFC3339Nano)},
			},
			res: []model.LabelSet{
				{
					"__name__": "up",
					"job":      "prometheus",
					"instance": "localhost:9090",
				},
			},
			warnings: []string{"a"},
		},

		{
			do:        doSeries("up", testTime.Add(-time.Minute), testTime),
			inErr:     fmt.Errorf("some error"),
			reqMethod: "GET",
			reqPath:   "/api/v1/series",
			reqParam: url.Values{
				"match": []string{"up"},
				"start": []string{testTime.Add(-time.Minute).Format(time.RFC3339Nano)},
				"end":   []string{testTime.Format(time.RFC3339Nano)},
			},
			err: fmt.Errorf("some error"),
		},
		// Series with error and warning.
		{
			do:         doSeries("up", testTime.Add(-time.Minute), testTime),
			inErr:      fmt.Errorf("some error"),
			inWarnings: []string{"a"},
			reqMethod:  "GET",
			reqPath:    "/api/v1/series",
			reqParam: url.Values{
				"match": []string{"up"},
				"start": []string{testTime.Add(-time.Minute).Format(time.RFC3339Nano)},
				"end":   []string{testTime.Format(time.RFC3339Nano)},
			},
			err:      fmt.Errorf("some error"),
			warnings: []string{"a"},
		},

		{
			do: doSnapshot(true),
			inRes: map[string]string{
				"name": "20171210T211224Z-2be650b6d019eb54",
			},
			reqMethod: "POST",
			reqPath:   "/api/v1/admin/tsdb/snapshot",
			reqParam: url.Values{
				"skip_head": []string{"true"},
			},
			res: SnapshotResult{
				Name: "20171210T211224Z-2be650b6d019eb54",
			},
		},

		{
			do:        doSnapshot(true),
			inErr:     fmt.Errorf("some error"),
			reqMethod: "POST",
			reqPath:   "/api/v1/admin/tsdb/snapshot",
			err:       fmt.Errorf("some error"),
		},

		{
			do:        doCleanTombstones(),
			reqMethod: "POST",
			reqPath:   "/api/v1/admin/tsdb/clean_tombstones",
		},

		{
			do:        doCleanTombstones(),
			inErr:     fmt.Errorf("some error"),
			reqMethod: "POST",
			reqPath:   "/api/v1/admin/tsdb/clean_tombstones",
			err:       fmt.Errorf("some error"),
		},

		{
			do: doDeleteSeries("up", testTime.Add(-time.Minute), testTime),
			inRes: []map[string]string{
				{
					"__name__": "up",
					"job":      "prometheus",
					"instance": "localhost:9090"},
			},
			reqMethod: "POST",
			reqPath:   "/api/v1/admin/tsdb/delete_series",
			reqParam: url.Values{
				"match": []string{"up"},
				"start": []string{testTime.Add(-time.Minute).Format(time.RFC3339Nano)},
				"end":   []string{testTime.Format(time.RFC3339Nano)},
			},
		},

		{
			do:        doDeleteSeries("up", testTime.Add(-time.Minute), testTime),
			inErr:     fmt.Errorf("some error"),
			reqMethod: "POST",
			reqPath:   "/api/v1/admin/tsdb/delete_series",
			reqParam: url.Values{
				"match": []string{"up"},
				"start": []string{testTime.Add(-time.Minute).Format(time.RFC3339Nano)},
				"end":   []string{testTime.Format(time.RFC3339Nano)},
			},
			err: fmt.Errorf("some error"),
		},

		{
			do:        doConfig(),
			reqMethod: "GET",
			reqPath:   "/api/v1/status/config",
			inRes: map[string]string{
				"yaml": "<content of the loaded config file in YAML>",
			},
			res: ConfigResult{
				YAML: "<content of the loaded config file in YAML>",
			},
		},

		{
			do:        doConfig(),
			reqMethod: "GET",
			reqPath:   "/api/v1/status/config",
			inErr:     fmt.Errorf("some error"),
			err:       fmt.Errorf("some error"),
		},

		{
			do:        doFlags(),
			reqMethod: "GET",
			reqPath:   "/api/v1/status/flags",
			inRes: map[string]string{
				"alertmanager.notification-queue-capacity": "10000",
				"alertmanager.timeout":                     "10s",
				"log.level":                                "info",
				"query.lookback-delta":                     "5m",
				"query.max-concurrency":                    "20",
			},
			res: FlagsResult{
				"alertmanager.notification-queue-capacity": "10000",
				"alertmanager.timeout":                     "10s",
				"log.level":                                "info",
				"query.lookback-delta":                     "5m",
				"query.max-concurrency":                    "20",
			},
		},

		{
			do:        doFlags(),
			reqMethod: "GET",
			reqPath:   "/api/v1/status/flags",
			inErr:     fmt.Errorf("some error"),
			err:       fmt.Errorf("some error"),
		},

		{
			do:        doRuntimeinfo(),
			reqMethod: "GET",
			reqPath:   "/api/v1/status/runtimeinfo",
			inErr:     fmt.Errorf("some error"),
			err:       fmt.Errorf("some error"),
		},

		{
			do:        doRuntimeinfo(),
			reqMethod: "GET",
			reqPath:   "/api/v1/status/runtimeinfo",
			inRes: map[string]interface{}{
				"startTime":           "2020-05-18T15:52:53.4503113Z",
				"CWD":                 "/prometheus",
				"reloadConfigSuccess": true,
				"lastConfigTime":      "2020-05-18T15:52:56Z",
				"chunkCount":          72692,
				"timeSeriesCount":     18476,
				"corruptionCount":     0,
				"goroutineCount":      217,
				"GOMAXPROCS":          2,
				"GOGC":                "100",
				"GODEBUG":             "allocfreetrace",
				"storageRetention":    "1d",
			},
			res: RuntimeinfoResult{
				StartTime:           "2020-05-18T15:52:53.4503113Z",
				CWD:                 "/prometheus",
				ReloadConfigSuccess: true,
				LastConfigTime:      "2020-05-18T15:52:56Z",
				ChunkCount:          72692,
				TimeSeriesCount:     18476,
				CorruptionCount:     0,
				GoroutineCount:      217,
				GOMAXPROCS:          2,
				GOGC:                "100",
				GODEBUG:             "allocfreetrace",
				StorageRetention:    "1d",
			},
		},

		{
			do:        doAlertManagers(),
			reqMethod: "GET",
			reqPath:   "/api/v1/alertmanagers",
			inRes: map[string]interface{}{
				"activeAlertManagers": []map[string]string{
					{
						"url": "http://127.0.0.1:9091/api/v1/alerts",
					},
				},
				"droppedAlertManagers": []map[string]string{
					{
						"url": "http://127.0.0.1:9092/api/v1/alerts",
					},
				},
			},
			res: AlertManagersResult{
				Active: []AlertManager{
					{
						URL: "http://127.0.0.1:9091/api/v1/alerts",
					},
				},
				Dropped: []AlertManager{
					{
						URL: "http://127.0.0.1:9092/api/v1/alerts",
					},
				},
			},
		},

		{
			do:        doAlertManagers(),
			reqMethod: "GET",
			reqPath:   "/api/v1/alertmanagers",
			inErr:     fmt.Errorf("some error"),
			err:       fmt.Errorf("some error"),
		},

		{
			do:        doRules(),
			reqMethod: "GET",
			reqPath:   "/api/v1/rules",
			inRes: map[string]interface{}{
				"groups": []map[string]interface{}{
					{
						"file":     "/rules.yaml",
						"interval": 60,
						"name":     "example",
						"rules": []map[string]interface{}{
							{
								"alerts": []map[string]interface{}{
									{
										"activeAt": testTime.UTC().Format(time.RFC3339Nano),
										"annotations": map[string]interface{}{
											"summary": "High request latency",
										},
										"labels": map[string]interface{}{
											"alertname": "HighRequestLatency",
											"severity":  "page",
										},
										"state": "firing",
										"value": "1e+00",
									},
								},
								"annotations": map[string]interface{}{
									"summary": "High request latency",
								},
								"duration": 600,
								"health":   "ok",
								"labels": map[string]interface{}{
									"severity": "page",
								},
								"name":  "HighRequestLatency",
								"query": "job:request_latency_seconds:mean5m{job=\"myjob\"} > 0.5",
								"type":  "alerting",
							},
							{
								"health": "ok",
								"name":   "job:http_inprogress_requests:sum",
								"query":  "sum(http_inprogress_requests) by (job)",
								"type":   "recording",
							},
						},
					},
				},
			},
			res: RulesResult{
				Groups: []RuleGroup{
					{
						Name:     "example",
						File:     "/rules.yaml",
						Interval: 60,
						Rules: []interface{}{
							AlertingRule{
								Alerts: []*Alert{
									{
										ActiveAt: testTime.UTC(),
										Annotations: model.LabelSet{
											"summary": "High request latency",
										},
										Labels: model.LabelSet{
											"alertname": "HighRequestLatency",
											"severity":  "page",
										},
										State: AlertStateFiring,
										Value: "1e+00",
									},
								},
								Annotations: model.LabelSet{
									"summary": "High request latency",
								},
								Labels: model.LabelSet{
									"severity": "page",
								},
								Duration:  600,
								Health:    RuleHealthGood,
								Name:      "HighRequestLatency",
								Query:     "job:request_latency_seconds:mean5m{job=\"myjob\"} > 0.5",
								LastError: "",
							},
							RecordingRule{
								Health:    RuleHealthGood,
								Name:      "job:http_inprogress_requests:sum",
								Query:     "sum(http_inprogress_requests) by (job)",
								LastError: "",
							},
						},
					},
				},
			},
		},

		{
			do:        doRules(),
			reqMethod: "GET",
			reqPath:   "/api/v1/rules",
			inErr:     fmt.Errorf("some error"),
			err:       fmt.Errorf("some error"),
		},

		{
			do:        doTargets(),
			reqMethod: "GET",
			reqPath:   "/api/v1/targets",
			inRes: map[string]interface{}{
				"activeTargets": []map[string]interface{}{
					{
						"discoveredLabels": map[string]string{
							"__address__":      "127.0.0.1:9090",
							"__metrics_path__": "/metrics",
							"__scheme__":       "http",
							"job":              "prometheus",
						},
						"labels": map[string]string{
							"instance": "127.0.0.1:9090",
							"job":      "prometheus",
						},
						"scrapeUrl":  "http://127.0.0.1:9090",
						"lastError":  "error while scraping target",
						"lastScrape": testTime.UTC().Format(time.RFC3339Nano),
						"health":     "up",
					},
				},
				"droppedTargets": []map[string]interface{}{
					{
						"discoveredLabels": map[string]string{
							"__address__":      "127.0.0.1:9100",
							"__metrics_path__": "/metrics",
							"__scheme__":       "http",
							"job":              "node",
						},
					},
				},
			},
			res: TargetsResult{
				Active: []ActiveTarget{
					{
						DiscoveredLabels: map[string]string{
							"__address__":      "127.0.0.1:9090",
							"__metrics_path__": "/metrics",
							"__scheme__":       "http",
							"job":              "prometheus",
						},
						Labels: model.LabelSet{
							"instance": "127.0.0.1:9090",
							"job":      "prometheus",
						},
						ScrapeURL:  "http://127.0.0.1:9090",
						LastError:  "error while scraping target",
						LastScrape: testTime.UTC(),
						Health:     HealthGood,
					},
				},
				Dropped: []DroppedTarget{
					{
						DiscoveredLabels: map[string]string{
							"__address__":      "127.0.0.1:9100",
							"__metrics_path__": "/metrics",
							"__scheme__":       "http",
							"job":              "node",
						},
					},
				},
			},
		},

		{
			do:        doTargets(),
			reqMethod: "GET",
			reqPath:   "/api/v1/targets",
			inErr:     fmt.Errorf("some error"),
			err:       fmt.Errorf("some error"),
		},

		{
			do: doTargetsMetadata("{job=\"prometheus\"}", "go_goroutines", "1"),
			inRes: []map[string]interface{}{
				{
					"target": map[string]interface{}{
						"instance": "127.0.0.1:9090",
						"job":      "prometheus",
					},
					"type": "gauge",
					"help": "Number of goroutines that currently exist.",
					"unit": "",
				},
			},
			reqMethod: "GET",
			reqPath:   "/api/v1/targets/metadata",
			reqParam: url.Values{
				"match_target": []string{"{job=\"prometheus\"}"},
				"metric":       []string{"go_goroutines"},
				"limit":        []string{"1"},
			},
			res: []MetricMetadata{
				{
					Target: map[string]string{
						"instance": "127.0.0.1:9090",
						"job":      "prometheus",
					},
					Type: "gauge",
					Help: "Number of goroutines that currently exist.",
					Unit: "",
				},
			},
		},

		{
			do:        doTargetsMetadata("{job=\"prometheus\"}", "go_goroutines", "1"),
			inErr:     fmt.Errorf("some error"),
			reqMethod: "GET",
			reqPath:   "/api/v1/targets/metadata",
			reqParam: url.Values{
				"match_target": []string{"{job=\"prometheus\"}"},
				"metric":       []string{"go_goroutines"},
				"limit":        []string{"1"},
			},
			err: fmt.Errorf("some error"),
		},

		{
			do: doMetadata("go_goroutines", "1"),
			inRes: map[string]interface{}{
				"go_goroutines": []map[string]interface{}{
					{
						"type": "gauge",
						"help": "Number of goroutines that currently exist.",
						"unit": "",
					},
				},
			},
			reqMethod: "GET",
			reqPath:   "/api/v1/metadata",
			reqParam: url.Values{
				"metric": []string{"go_goroutines"},
				"limit":  []string{"1"},
			},
			res: map[string][]Metadata{
				"go_goroutines": []Metadata{
					{
						Type: "gauge",
						Help: "Number of goroutines that currently exist.",
						Unit: "",
					},
				},
			},
		},

		{
			do:        doMetadata("", "1"),
			inErr:     fmt.Errorf("some error"),
			reqMethod: "GET",
			reqPath:   "/api/v1/metadata",
			reqParam: url.Values{
				"metric": []string{""},
				"limit":  []string{"1"},
			},
			err: fmt.Errorf("some error"),
		},
	}

	var tests []apiTest
	tests = append(tests, queryTests...)

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			tc.curTest = test

			res, warnings, err := test.do()

			if (test.inWarnings == nil) != (warnings == nil) && !reflect.DeepEqual(test.inWarnings, warnings) {
				t.Fatalf("mismatch in warnings expected=%v actual=%v", test.inWarnings, warnings)
			}

			if test.err != nil {
				if err == nil {
					t.Fatalf("expected error %q but got none", test.err)
				}
				if err.Error() != test.err.Error() {
					t.Errorf("unexpected error: want %s, got %s", test.err, err)
				}
				if apiErr, ok := err.(*Error); ok {
					if apiErr.Detail != test.inRes {
						t.Errorf("%q should be %q", apiErr.Detail, test.inRes)
					}
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %s", err)
			}

			if !reflect.DeepEqual(res, test.res) {
				t.Errorf("unexpected result: want %v, got %v", test.res, res)
			}
		})
	}
}

type testClient struct {
	*testing.T

	ch  chan apiClientTest
	req *http.Request
}

type apiClientTest struct {
	code             int
	response         interface{}
	expectedBody     string
	expectedErr      *Error
	expectedWarnings Warnings
}

func (c *testClient) URL(ep string, args map[string]string) *url.URL {
	return nil
}

func (c *testClient) Do(ctx context.Context, req *http.Request) (*http.Response, []byte, error) {
	if ctx == nil {
		c.Fatalf("context was not passed down")
	}
	if req != c.req {
		c.Fatalf("request was not passed down")
	}

	test := <-c.ch

	var b []byte
	var err error

	switch v := test.response.(type) {
	case string:
		b = []byte(v)
	default:
		b, err = json.Marshal(v)
		if err != nil {
			c.Fatal(err)
		}
	}

	resp := &http.Response{
		StatusCode: test.code,
	}

	return resp, b, nil
}

func TestAPIClientDo(t *testing.T) {
	tests := []apiClientTest{
		{
			code: statusAPIError,
			response: &apiResponse{
				Status:    "error",
				Data:      json.RawMessage(`null`),
				ErrorType: ErrBadData,
				Error:     "failed",
			},
			expectedErr: &Error{
				Type: ErrBadData,
				Msg:  "failed",
			},
			expectedBody: `null`,
		},
		{
			code: statusAPIError,
			response: &apiResponse{
				Status:    "error",
				Data:      json.RawMessage(`"test"`),
				ErrorType: ErrTimeout,
				Error:     "timed out",
			},
			expectedErr: &Error{
				Type: ErrTimeout,
				Msg:  "timed out",
			},
			expectedBody: `test`,
		},
		{
			code:     http.StatusInternalServerError,
			response: "500 error details",
			expectedErr: &Error{
				Type:   ErrServer,
				Msg:    "server error: 500",
				Detail: "500 error details",
			},
		},
		{
			code:     http.StatusNotFound,
			response: "404 error details",
			expectedErr: &Error{
				Type:   ErrClient,
				Msg:    "client error: 404",
				Detail: "404 error details",
			},
		},
		{
			code: http.StatusBadRequest,
			response: &apiResponse{
				Status:    "error",
				Data:      json.RawMessage(`null`),
				ErrorType: ErrBadData,
				Error:     "end timestamp must not be before start time",
			},
			expectedErr: &Error{
				Type: ErrBadData,
				Msg:  "end timestamp must not be before start time",
			},
		},
		{
			code:     statusAPIError,
			response: "bad json",
			expectedErr: &Error{
				Type: ErrBadResponse,
				Msg:  "readObjectStart: expect { or n, but found b, error found in #1 byte of ...|bad json|..., bigger context ...|bad json|...",
			},
		},
		{
			code: statusAPIError,
			response: &apiResponse{
				Status: "success",
				Data:   json.RawMessage(`"test"`),
			},
			expectedErr: &Error{
				Type: ErrBadResponse,
				Msg:  "inconsistent body for response code",
			},
		},
		{
			code: statusAPIError,
			response: &apiResponse{
				Status:    "success",
				Data:      json.RawMessage(`"test"`),
				ErrorType: ErrTimeout,
				Error:     "timed out",
			},
			expectedErr: &Error{
				Type: ErrBadResponse,
				Msg:  "inconsistent body for response code",
			},
		},
		{
			code: http.StatusOK,
			response: &apiResponse{
				Status:    "error",
				Data:      json.RawMessage(`"test"`),
				ErrorType: ErrTimeout,
				Error:     "timed out",
			},
			expectedErr: &Error{
				Type: ErrTimeout,
				Msg:  "timed out",
			},
		},
		{
			code: http.StatusOK,
			response: &apiResponse{
				Status:    "error",
				Data:      json.RawMessage(`"test"`),
				ErrorType: ErrTimeout,
				Error:     "timed out",
				Warnings:  []string{"a"},
			},
			expectedErr: &Error{
				Type: ErrTimeout,
				Msg:  "timed out",
			},
			expectedWarnings: []string{"a"},
		},
	}

	tc := &testClient{
		T:   t,
		ch:  make(chan apiClientTest, 1),
		req: &http.Request{},
	}
	client := &apiClientImpl{
		client: tc,
	}

	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {

			tc.ch <- test

			_, body, warnings, err := client.Do(context.Background(), tc.req)

			if test.expectedWarnings != nil {
				if !reflect.DeepEqual(test.expectedWarnings, warnings) {
					t.Fatalf("mismatch in warnings expected=%v actual=%v", test.expectedWarnings, warnings)
				}
			} else {
				if warnings != nil {
					t.Fatalf("unexpexted warnings: %v", warnings)
				}
			}

			if test.expectedErr != nil {
				if err == nil {
					t.Fatal("expected error, but got none")
				}

				if test.expectedErr.Error() != err.Error() {
					t.Fatalf("expected error:%v, but got:%v", test.expectedErr.Error(), err.Error())
				}

				if test.expectedErr.Detail != "" {
					apiErr := err.(*Error)
					if apiErr.Detail != test.expectedErr.Detail {
						t.Fatalf("expected error detail :%v, but got:%v", apiErr.Detail, test.expectedErr.Detail)
					}
				}

				return
			}

			if err != nil {
				t.Fatalf("unexpected error:%v", err)
			}
			if test.expectedBody != string(body) {
				t.Fatalf("expected body :%v, but got:%v", test.expectedBody, string(body))
			}
		})

	}
}

func TestSamplesJsonSerialization(t *testing.T) {
	tests := []struct {
		point    model.SamplePair
		expected string
	}{
		{
			point:    model.SamplePair{0, 0},
			expected: `[0,"0"]`,
		},
		{
			point:    model.SamplePair{1, 20},
			expected: `[0.001,"20"]`,
		},
		{
			point:    model.SamplePair{10, 20},
			expected: `[0.010,"20"]`,
		},
		{
			point:    model.SamplePair{100, 20},
			expected: `[0.100,"20"]`,
		},
		{
			point:    model.SamplePair{1001, 20},
			expected: `[1.001,"20"]`,
		},
		{
			point:    model.SamplePair{1010, 20},
			expected: `[1.010,"20"]`,
		},
		{
			point:    model.SamplePair{1100, 20},
			expected: `[1.100,"20"]`,
		},
		{
			point:    model.SamplePair{12345678123456555, 20},
			expected: `[12345678123456.555,"20"]`,
		},
		{
			point:    model.SamplePair{-1, 20},
			expected: `[-0.001,"20"]`,
		},
		{
			point:    model.SamplePair{0, model.SampleValue(math.NaN())},
			expected: `[0,"NaN"]`,
		},
		{
			point:    model.SamplePair{0, model.SampleValue(math.Inf(1))},
			expected: `[0,"+Inf"]`,
		},
		{
			point:    model.SamplePair{0, model.SampleValue(math.Inf(-1))},
			expected: `[0,"-Inf"]`,
		},
		{
			point:    model.SamplePair{0, model.SampleValue(1.2345678e6)},
			expected: `[0,"1234567.8"]`,
		},
		{
			point:    model.SamplePair{0, 1.2345678e-6},
			expected: `[0,"0.0000012345678"]`,
		},
		{
			point:    model.SamplePair{0, 1.2345678e-67},
			expected: `[0,"1.2345678e-67"]`,
		},
	}

	for _, test := range tests {
		t.Run(test.expected, func(t *testing.T) {
			b, err := json.Marshal(test.point)
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != test.expected {
				t.Fatalf("Mismatch marshal expected=%s actual=%s", test.expected, string(b))
			}

			// To test Unmarshal we will Unmarshal then re-Marshal this way we
			// can do a string compare, otherwise Nan values don't show equivalence
			// properly.
			var sp model.SamplePair
			if err = json.Unmarshal(b, &sp); err != nil {
				t.Fatal(err)
			}

			b, err = json.Marshal(sp)
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != test.expected {
				t.Fatalf("Mismatch marshal expected=%s actual=%s", test.expected, string(b))
			}
		})
	}
}

type httpTestClient struct {
	client http.Client
}

func (c *httpTestClient) URL(ep string, args map[string]string) *url.URL {
	return nil
}

func (c *httpTestClient) Do(ctx context.Context, req *http.Request) (*http.Response, []byte, error) {
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, nil, err
	}

	var body []byte
	done := make(chan struct{})
	go func() {
		body, err = ioutil.ReadAll(resp.Body)
		close(done)
	}()

	select {
	case <-ctx.Done():
		<-done
		err = resp.Body.Close()
		if err == nil {
			err = ctx.Err()
		}
	case <-done:
	}

	return resp, body, err
}

func TestDoGetFallback(t *testing.T) {
	v := url.Values{"a": []string{"1", "2"}}

	type testResponse struct {
		Values string
		Method string
	}

	// Start a local HTTP server.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		req.ParseForm()
		testResp, _ := json.Marshal(&testResponse{
			Values: req.Form.Encode(),
			Method: req.Method,
		})

		apiResp := &apiResponse{
			Data: testResp,
		}

		body, _ := json.Marshal(apiResp)

		if req.Method == http.MethodPost {
			if req.URL.Path == "/blockPost" {
				http.Error(w, string(body), http.StatusMethodNotAllowed)
				return
			}
		}

		w.Write(body)
	}))
	// Close the server when test finishes.
	defer server.Close()

	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	client := &httpTestClient{client: *(server.Client())}
	api := &apiClientImpl{
		client: client,
	}

	// Do a post, and ensure that the post succeeds.
	_, b, _, err := api.DoGetFallback(context.TODO(), u, v)
	if err != nil {
		t.Fatalf("Error doing local request: %v", err)
	}
	resp := &testResponse{}
	if err := json.Unmarshal(b, resp); err != nil {
		t.Fatal(err)
	}
	if resp.Method != http.MethodPost {
		t.Fatalf("Mismatch method")
	}
	if resp.Values != v.Encode() {
		t.Fatalf("Mismatch in values")
	}

	// Do a fallbcak to a get.
	u.Path = "/blockPost"
	_, b, _, err = api.DoGetFallback(context.TODO(), u, v)
	if err != nil {
		t.Fatalf("Error doing local request: %v", err)
	}
	if err := json.Unmarshal(b, resp); err != nil {
		t.Fatal(err)
	}
	if resp.Method != http.MethodGet {
		t.Fatalf("Mismatch method")
	}
	if resp.Values != v.Encode() {
		t.Fatalf("Mismatch in values")
	}
}
