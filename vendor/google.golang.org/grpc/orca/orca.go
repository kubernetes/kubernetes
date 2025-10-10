/*
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
 */

// Package orca implements Open Request Cost Aggregation, which is an open
// standard for request cost aggregation and reporting by backends and the
// corresponding aggregation of such reports by L7 load balancers (such as
// Envoy) on the data plane. In a proxyless world with gRPC enabled
// applications, aggregation of such reports will be done by the gRPC client.
//
// # Experimental
//
// Notice: All APIs is this package are EXPERIMENTAL and may be changed or
// removed in a later release.
package orca

import (
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/balancerload"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/orca/internal"
)

var logger = grpclog.Component("orca-backend-metrics")

// loadParser implements the Parser interface defined in `internal/balancerload`
// package. This interface is used by the client stream to parse load reports
// sent by the server in trailer metadata. The parsed loads are then sent to
// balancers via balancer.DoneInfo.
//
// The grpc package cannot directly call toLoadReport() as that would cause an
// import cycle. Hence this roundabout method is used.
type loadParser struct{}

func (loadParser) Parse(md metadata.MD) any {
	lr, err := internal.ToLoadReport(md)
	if err != nil {
		logger.Infof("Parse failed: %v", err)
	}
	return lr
}

func init() {
	balancerload.SetParser(loadParser{})
}
