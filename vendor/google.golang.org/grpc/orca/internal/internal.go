/*
 *
 * Copyright 2022 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package internal contains orca-internal code, for testing purposes and to
// avoid polluting the godoc of the top-level orca package.
package internal

import (
	"errors"
	"fmt"

	ibackoff "google.golang.org/grpc/internal/backoff"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/proto"

	v3orcapb "github.com/cncf/xds/go/xds/data/orca/v3"
)

// AllowAnyMinReportingInterval prevents clamping of the MinReportingInterval
// configured via ServiceOptions, to a minimum of 30s.
//
// For testing purposes only.
var AllowAnyMinReportingInterval any // func(*ServiceOptions)

// DefaultBackoffFunc is used by the producer to control its backoff behavior.
//
// For testing purposes only.
var DefaultBackoffFunc = ibackoff.DefaultExponential.Backoff

// TrailerMetadataKey is the key in which the per-call backend metrics are
// transmitted.
const TrailerMetadataKey = "endpoint-load-metrics-bin"

// ToLoadReport unmarshals a binary encoded [ORCA LoadReport] protobuf message
// from md and returns the corresponding struct. The load report is expected to
// be stored as the value for key "endpoint-load-metrics-bin".
//
// If no load report was found in the provided metadata, if multiple load
// reports are found, or if the load report found cannot be parsed, an error is
// returned.
//
// [ORCA LoadReport]: (https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto#L15)
func ToLoadReport(md metadata.MD) (*v3orcapb.OrcaLoadReport, error) {
	vs := md.Get(TrailerMetadataKey)
	if len(vs) == 0 {
		return nil, nil
	}
	if len(vs) != 1 {
		return nil, errors.New("multiple orca load reports found in provided metadata")
	}
	ret := new(v3orcapb.OrcaLoadReport)
	if err := proto.Unmarshal([]byte(vs[0]), ret); err != nil {
		return nil, fmt.Errorf("failed to unmarshal load report found in metadata: %v", err)
	}
	return ret, nil
}
