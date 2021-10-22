// Copyright 2017, OpenCensus Authors
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

package ocgrpc

import (
	"testing"

	"context"
	"go.opencensus.io/trace"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/stats"
)

func TestClientHandler_traceTagRPC(t *testing.T) {
	ch := &ClientHandler{}
	ch.StartOptions.Sampler = trace.AlwaysSample()
	rti := &stats.RPCTagInfo{
		FullMethodName: "xxx",
	}
	ctx := context.Background()
	ctx = ch.traceTagRPC(ctx, rti)

	span := trace.FromContext(ctx)
	if span == nil {
		t.Fatal("expected span, got nil")
	}
	if !span.IsRecordingEvents() {
		t.Errorf("span should be sampled")
	}
	md, ok := metadata.FromOutgoingContext(ctx)
	if !ok || len(md) == 0 || len(md[traceContextKey]) == 0 {
		t.Fatal("no metadata")
	}
}
