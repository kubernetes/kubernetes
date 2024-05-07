// Copyright 2015 The Prometheus Authors
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

// Package expfmt contains tools for reading and writing Prometheus metrics.
package expfmt

// Format specifies the HTTP content type of the different wire protocols.
type Format string

// Constants to assemble the Content-Type values for the different wire protocols.
const (
	TextVersion              = "0.0.4"
	ProtoType                = `application/vnd.google.protobuf`
	ProtoProtocol            = `io.prometheus.client.MetricFamily`
	ProtoFmt                 = ProtoType + "; proto=" + ProtoProtocol + ";"
	OpenMetricsType          = `application/openmetrics-text`
	OpenMetricsVersion_0_0_1 = "0.0.1"
	OpenMetricsVersion_1_0_0 = "1.0.0"

	// The Content-Type values for the different wire protocols.
	FmtUnknown           Format = `<unknown>`
	FmtText              Format = `text/plain; version=` + TextVersion + `; charset=utf-8`
	FmtProtoDelim        Format = ProtoFmt + ` encoding=delimited`
	FmtProtoText         Format = ProtoFmt + ` encoding=text`
	FmtProtoCompact      Format = ProtoFmt + ` encoding=compact-text`
	FmtOpenMetrics_1_0_0 Format = OpenMetricsType + `; version=` + OpenMetricsVersion_1_0_0 + `; charset=utf-8`
	FmtOpenMetrics_0_0_1 Format = OpenMetricsType + `; version=` + OpenMetricsVersion_0_0_1 + `; charset=utf-8`
)

const (
	hdrContentType = "Content-Type"
	hdrAccept      = "Accept"
)
