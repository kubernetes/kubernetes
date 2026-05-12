/*
 *
 * Copyright 2025 gRPC authors.
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

package stats

import (
	"context"

	"google.golang.org/grpc/stats"
)

type combinedHandler struct {
	handlers []stats.Handler
}

// NewCombinedHandler combines multiple stats.Handlers into a single handler.
//
// It returns nil if no handlers are provided. If only one handler is
// provided, it is returned directly without wrapping.
func NewCombinedHandler(handlers ...stats.Handler) stats.Handler {
	switch len(handlers) {
	case 0:
		return nil
	case 1:
		return handlers[0]
	default:
		return &combinedHandler{handlers: handlers}
	}
}

func (ch *combinedHandler) TagRPC(ctx context.Context, info *stats.RPCTagInfo) context.Context {
	for _, h := range ch.handlers {
		ctx = h.TagRPC(ctx, info)
	}
	return ctx
}

func (ch *combinedHandler) HandleRPC(ctx context.Context, stats stats.RPCStats) {
	for _, h := range ch.handlers {
		h.HandleRPC(ctx, stats)
	}
}

func (ch *combinedHandler) TagConn(ctx context.Context, info *stats.ConnTagInfo) context.Context {
	for _, h := range ch.handlers {
		ctx = h.TagConn(ctx, info)
	}
	return ctx
}

func (ch *combinedHandler) HandleConn(ctx context.Context, stats stats.ConnStats) {
	for _, h := range ch.handlers {
		h.HandleConn(ctx, stats)
	}
}
