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

package metrics

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestFetchMetrics(t *testing.T) {
	testCases := []struct {
		desc    string
		metrics []string
		want    string
		emit    func()
	}{
		{
			desc: "basic test",
			metrics: []string{
				"apiserver_externaljwt_fetch_keys_request_total",
			},
			emit: func() {
				RecordFetchKeysAttempt(status.New(codes.Internal, "error ocured").Err())
			},
			want: fmt.Sprintf(`
			# HELP apiserver_externaljwt_fetch_keys_request_total [ALPHA] Total attempts at syncing supported JWKs
			# TYPE apiserver_externaljwt_fetch_keys_request_total counter
			apiserver_externaljwt_fetch_keys_request_total{code="%s"} 1
			`, codes.Internal.String()),
		},
		{
			desc: "wrapped error code",
			metrics: []string{
				"apiserver_externaljwt_fetch_keys_request_total",
			},
			emit: func() {
				RecordFetchKeysAttempt(fmt.Errorf("some error %w", status.New(codes.Canceled, "error ocured").Err()))
			},
			want: fmt.Sprintf(`
			# HELP apiserver_externaljwt_fetch_keys_request_total [ALPHA] Total attempts at syncing supported JWKs
			# TYPE apiserver_externaljwt_fetch_keys_request_total counter
			apiserver_externaljwt_fetch_keys_request_total{code="%s"} 1
			apiserver_externaljwt_fetch_keys_request_total{code="%s"} 1
			`, codes.Internal.String(), codes.Canceled.String()),
		},
		{
			desc: "success count appears",
			metrics: []string{
				"apiserver_externaljwt_fetch_keys_request_total",
			},
			emit: func() {
				RecordFetchKeysAttempt(nil)
			},
			want: fmt.Sprintf(`
			# HELP apiserver_externaljwt_fetch_keys_request_total [ALPHA] Total attempts at syncing supported JWKs
			# TYPE apiserver_externaljwt_fetch_keys_request_total counter
			apiserver_externaljwt_fetch_keys_request_total{code="%s"} 1
			apiserver_externaljwt_fetch_keys_request_total{code="%s"} 1
			apiserver_externaljwt_fetch_keys_request_total{code="%s"} 1
			`, codes.Internal.String(), codes.Canceled.String(), codes.OK.String()),
		},
		{
			desc: "success count increments",
			metrics: []string{
				"apiserver_externaljwt_fetch_keys_request_total",
			},
			emit: func() {
				RecordFetchKeysAttempt(nil)
			},
			want: fmt.Sprintf(`
			# HELP apiserver_externaljwt_fetch_keys_request_total [ALPHA] Total attempts at syncing supported JWKs
			# TYPE apiserver_externaljwt_fetch_keys_request_total counter
			apiserver_externaljwt_fetch_keys_request_total{code="%s"} 1
			apiserver_externaljwt_fetch_keys_request_total{code="%s"} 1
			apiserver_externaljwt_fetch_keys_request_total{code="%s"} 2
			`, codes.Internal.String(), codes.Canceled.String(), codes.OK.String()),
		},
	}

	RegisterMetrics()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			tt.emit()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestTokenGenMetrics(t *testing.T) {
	testCases := []struct {
		desc    string
		metrics []string
		want    string
		emit    func()
	}{
		{
			desc: "basic test",
			metrics: []string{
				"apiserver_externaljwt_sign_request_total",
			},
			emit: func() {
				RecordTokenGenAttempt(fmt.Errorf("some error %w", status.New(codes.Internal, "error ocured").Err()))
			},
			want: fmt.Sprintf(`
			# HELP apiserver_externaljwt_sign_request_total [ALPHA] Total attempts at signing JWT
			# TYPE apiserver_externaljwt_sign_request_total counter
			apiserver_externaljwt_sign_request_total{code="%s"} 1
			`, codes.Internal.String()),
		},
		{
			desc: "wrapped error code",
			metrics: []string{
				"apiserver_externaljwt_sign_request_total",
			},
			emit: func() {
				RecordTokenGenAttempt(fmt.Errorf("some error %w", fmt.Errorf("some error %w", status.New(codes.Canceled, "error ocured").Err())))
			},
			want: fmt.Sprintf(`
			# HELP apiserver_externaljwt_sign_request_total [ALPHA] Total attempts at signing JWT
			# TYPE apiserver_externaljwt_sign_request_total counter
			apiserver_externaljwt_sign_request_total{code="%s"} 1
			apiserver_externaljwt_sign_request_total{code="%s"} 1
			`, codes.Internal.String(), codes.Canceled.String()),
		},
		{
			desc: "success count appears",
			metrics: []string{
				"apiserver_externaljwt_sign_request_total",
			},
			emit: func() {
				RecordTokenGenAttempt(nil)
			},
			want: fmt.Sprintf(`
			# HELP apiserver_externaljwt_sign_request_total [ALPHA] Total attempts at signing JWT
			# TYPE apiserver_externaljwt_sign_request_total counter
			apiserver_externaljwt_sign_request_total{code="%s"} 1
			apiserver_externaljwt_sign_request_total{code="%s"} 1
			apiserver_externaljwt_sign_request_total{code="%s"} 1
			`, codes.Internal.String(), codes.Canceled.String(), codes.OK.String()),
		},
		{
			desc: "success count increments",
			metrics: []string{
				"apiserver_externaljwt_sign_request_total",
			},
			emit: func() {
				RecordTokenGenAttempt(nil)
			},
			want: fmt.Sprintf(`
			# HELP apiserver_externaljwt_sign_request_total [ALPHA] Total attempts at signing JWT
			# TYPE apiserver_externaljwt_sign_request_total counter
			apiserver_externaljwt_sign_request_total{code="%s"} 1
			apiserver_externaljwt_sign_request_total{code="%s"} 1
			apiserver_externaljwt_sign_request_total{code="%s"} 2
			`, codes.Internal.String(), codes.Canceled.String(), codes.OK.String()),
		},
	}

	RegisterMetrics()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			tt.emit()
			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), tt.metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRecordKeyDataTimeStamp(t *testing.T) {

	dataTimeStamp1 := float64(time.Now().Unix())
	dataTimeStamp2 := float64(time.Now().Add(time.Second * 1200).Unix())

	testCases := []struct {
		desc    string
		metrics []string
		want    float64
		emit    func()
	}{
		{
			desc: "basic test",
			metrics: []string{
				"fetch_keys_data_timestamp",
			},
			emit: func() {
				RecordKeyDataTimeStamp(dataTimeStamp1)
			},
			want: dataTimeStamp1,
		},
		{
			desc: "update to a new value",
			metrics: []string{
				"fetch_keys_data_timestamp",
			},
			emit: func() {
				RecordKeyDataTimeStamp(dataTimeStamp1)
				RecordKeyDataTimeStamp(dataTimeStamp2)
			},
			want: dataTimeStamp2,
		},
	}

	RegisterMetrics()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			tt.emit()
			actualValue, err := testutil.GetGaugeMetricValue(dataTimeStamp.WithLabelValues())
			if err != nil {
				t.Errorf("error when getting gauge value for dataTimeStamp: %v", err)
			}
			if actualValue != float64(tt.want) {
				t.Errorf("Expected dataTimeStamp to be %v, got %v", tt.want, actualValue)
			}
		})
	}
}
