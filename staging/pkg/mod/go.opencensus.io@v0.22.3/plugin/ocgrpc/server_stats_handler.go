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
//

package ocgrpc

import (
	"time"

	"context"

	"go.opencensus.io/tag"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/stats"
)

// statsTagRPC gets the metadata from gRPC context, extracts the encoded tags from
// it and creates a new tag.Map and puts them into the returned context.
func (h *ServerHandler) statsTagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	startTime := time.Now()
	if info == nil {
		if grpclog.V(2) {
			grpclog.Infof("opencensus: TagRPC called with nil info.")
		}
		return ctx
	}
	d := &rpcData{
		startTime: startTime,
		method:    info.FullMethodName,
	}
	propagated := h.extractPropagatedTags(ctx)
	ctx = tag.NewContext(ctx, propagated)
	ctx, _ = tag.New(ctx, tag.Upsert(KeyServerMethod, methodName(info.FullMethodName)))
	return context.WithValue(ctx, rpcDataKey, d)
}

// extractPropagatedTags creates a new tag map containing the tags extracted from the
// gRPC metadata.
func (h *ServerHandler) extractPropagatedTags(ctx context.Context) *tag.Map {
	buf := stats.Tags(ctx)
	if buf == nil {
		return nil
	}
	propagated, err := tag.Decode(buf)
	if err != nil {
		if grpclog.V(2) {
			grpclog.Warningf("opencensus: Failed to decode tags from gRPC metadata failed to decode: %v", err)
		}
		return nil
	}
	return propagated
}
