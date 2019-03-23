// Copyright 2017 Google Inc. All Rights Reserved.
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

//This code has been taken from containerd repo to avoid large library import
package containerd

import (
	"github.com/containerd/containerd/namespaces"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

type namespaceInterceptor struct {
	namespace string
}

func (ni namespaceInterceptor) unary(ctx context.Context, method string, req, reply interface{}, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
	_, ok := namespaces.Namespace(ctx)
	if !ok {
		ctx = namespaces.WithNamespace(ctx, ni.namespace)
	}
	return invoker(ctx, method, req, reply, cc, opts...)
}

func (ni namespaceInterceptor) stream(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
	_, ok := namespaces.Namespace(ctx)
	if !ok {
		ctx = namespaces.WithNamespace(ctx, ni.namespace)
	}
	return streamer(ctx, desc, cc, method, opts...)
}

func newNSInterceptors(ns string) (grpc.UnaryClientInterceptor, grpc.StreamClientInterceptor) {
	ni := namespaceInterceptor{
		namespace: ns,
	}
	return grpc.UnaryClientInterceptor(ni.unary), grpc.StreamClientInterceptor(ni.stream)
}
