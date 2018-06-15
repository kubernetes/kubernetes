/*
 *
 * Copyright 2017 gRPC authors.
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

// Package passthrough implements a pass-through resolver. It sends the target
// name without scheme back to gRPC as resolved address.
package passthrough

import "google.golang.org/grpc/resolver"

const scheme = "passthrough"

type passthroughBuilder struct{}

func (*passthroughBuilder) Build(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOption) (resolver.Resolver, error) {
	r := &passthroughResolver{
		target: target,
		cc:     cc,
	}
	r.start()
	return r, nil
}

func (*passthroughBuilder) Scheme() string {
	return scheme
}

type passthroughResolver struct {
	target resolver.Target
	cc     resolver.ClientConn
}

func (r *passthroughResolver) start() {
	r.cc.NewAddress([]resolver.Address{{Addr: r.target.Endpoint}})
}

func (*passthroughResolver) ResolveNow(o resolver.ResolveNowOption) {}

func (*passthroughResolver) Close() {}

func init() {
	resolver.Register(&passthroughBuilder{})
}
