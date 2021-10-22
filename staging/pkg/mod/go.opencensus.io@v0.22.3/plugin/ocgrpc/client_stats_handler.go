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
	"context"
	"time"

	"go.opencensus.io/tag"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/stats"
)

// statsTagRPC gets the tag.Map populated by the application code, serializes
// its tags into the GRPC metadata in order to be sent to the server.
func (h *ClientHandler) statsTagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	startTime := time.Now()
	if info == nil {
		if grpclog.V(2) {
			grpclog.Info("clientHandler.TagRPC called with nil info.")
		}
		return ctx
	}

	d := &rpcData{
		startTime: startTime,
		method:    info.FullMethodName,
	}
	ts := tag.FromContext(ctx)
	if ts != nil {
		encoded := tag.Encode(ts)
		ctx = stats.SetTags(ctx, encoded)
	}

	return context.WithValue(ctx, rpcDataKey, d)
}
