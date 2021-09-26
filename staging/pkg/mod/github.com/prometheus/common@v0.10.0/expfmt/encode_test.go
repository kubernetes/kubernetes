// Copyright 2018 The Prometheus Authors
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

package expfmt

import (
	"bytes"
	"github.com/golang/protobuf/proto"
	dto "github.com/prometheus/client_model/go"
	"net/http"
	"testing"
)

func TestNegotiate(t *testing.T) {
	acceptValuePrefix := "application/vnd.google.protobuf;proto=io.prometheus.client.MetricFamily"
	tests := []struct {
		name              string
		acceptHeaderValue string
		expectedFmt       string
	}{
		{
			name:              "delimited format",
			acceptHeaderValue: acceptValuePrefix + ";encoding=delimited",
			expectedFmt:       string(FmtProtoDelim),
		},
		{
			name:              "text format",
			acceptHeaderValue: acceptValuePrefix + ";encoding=text",
			expectedFmt:       string(FmtProtoText),
		},
		{
			name:              "compact text format",
			acceptHeaderValue: acceptValuePrefix + ";encoding=compact-text",
			expectedFmt:       string(FmtProtoCompact),
		},
		{
			name:              "plain text format",
			acceptHeaderValue: "text/plain;version=0.0.4",
			expectedFmt:       string(FmtText),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			h := http.Header{}
			h.Add(hdrAccept, test.acceptHeaderValue)
			actualFmt := string(Negotiate(h))
			if actualFmt != test.expectedFmt {
				t.Errorf("expected Negotiate to return format %s, but got %s instead", test.expectedFmt, actualFmt)
			}
		})
	}
}

func TestEncode(t *testing.T) {
	var buff bytes.Buffer
	delimEncoder := NewEncoder(&buff, FmtProtoDelim)
	metric := &dto.MetricFamily{
		Name: proto.String("foo_metric"),
		Type: dto.MetricType_UNTYPED.Enum(),
		Metric: []*dto.Metric{
			{
				Untyped: &dto.Untyped{
					Value: proto.Float64(1.234),
				},
			},
		},
	}

	err := delimEncoder.Encode(metric)
	if err != nil {
		t.Errorf("unexpected error during encode: %s", err.Error())
	}

	out := buff.Bytes()
	if len(out) == 0 {
		t.Errorf("expected the output bytes buffer to be non-empty")
	}

	buff.Reset()

	compactEncoder := NewEncoder(&buff, FmtProtoCompact)
	err = compactEncoder.Encode(metric)
	if err != nil {
		t.Errorf("unexpected error during encode: %s", err.Error())
	}

	out = buff.Bytes()
	if len(out) == 0 {
		t.Errorf("expected the output bytes buffer to be non-empty")
	}

	buff.Reset()

	protoTextEncoder := NewEncoder(&buff, FmtProtoText)
	err = protoTextEncoder.Encode(metric)
	if err != nil {
		t.Errorf("unexpected error during encode: %s", err.Error())
	}

	out = buff.Bytes()
	if len(out) == 0 {
		t.Errorf("expected the output bytes buffer to be non-empty")
	}

	buff.Reset()

	textEncoder := NewEncoder(&buff, FmtText)
	err = textEncoder.Encode(metric)
	if err != nil {
		t.Errorf("unexpected error during encode: %s", err.Error())
	}

	out = buff.Bytes()
	if len(out) == 0 {
		t.Errorf("expected the output bytes buffer to be non-empty")
	}

	expected := "# TYPE foo_metric untyped\n" +
		"foo_metric 1.234\n"

	if string(out) != expected {
		t.Errorf("expected TextEncoder to return %s, but got %s instead", expected, string(out))
	}
}
