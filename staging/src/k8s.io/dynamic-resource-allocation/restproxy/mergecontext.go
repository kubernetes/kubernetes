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

package restproxy

import (
	"context"

	"google.golang.org/grpc"
)

// MergeContexts creates a new context where cancellation is handled by the
// root context. The values stored by the value context are used as fallback if
// the root context doesn't have a certain value.
func MergeContexts(rootCtx, valueCtx context.Context) context.Context {
	return mergeCtx{
		Context:  rootCtx,
		valueCtx: valueCtx,
	}
}

type mergeCtx struct {
	context.Context
	valueCtx context.Context
}

func (m mergeCtx) Value(i interface{}) interface{} {
	if v := m.Context.Value(i); v != nil {
		return v
	}
	return m.valueCtx.Value(i)
}

// UnaryContextInterceptor injects values from the context into the context
// used by the call chain.
func UnaryContextInterceptor(valueCtx context.Context) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		ctx = MergeContexts(ctx, valueCtx)
		return handler(ctx, req)
	}
}

// StreamContextInterceptor does the same as UnaryContextInterceptor for streams.
func StreamContextInterceptor(valueCtx context.Context) grpc.StreamServerInterceptor {
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		ctx := MergeContexts(ss.Context(), valueCtx)
		return handler(srv, mergeServerStream{ServerStream: ss, ctx: ctx})
	}
}

type mergeServerStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (m mergeServerStream) Context() context.Context {
	return m.ctx
}
