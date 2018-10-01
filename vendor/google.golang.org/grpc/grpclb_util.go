/*
 *
 * Copyright 2016 gRPC authors.
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

package grpc

import (
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/resolver"
)

// The parent ClientConn should re-resolve when grpclb loses connection to the
// remote balancer. When the ClientConn inside grpclb gets a TransientFailure,
// it calls lbManualResolver.ResolveNow(), which calls parent ClientConn's
// ResolveNow, and eventually results in re-resolve happening in parent
// ClientConn's resolver (DNS for example).
//
//                          parent
//                          ClientConn
//  +-----------------------------------------------------------------+
//  |             parent          +---------------------------------+ |
//  | DNS         ClientConn      |  grpclb                         | |
//  | resolver    balancerWrapper |                                 | |
//  | +              +            |    grpclb          grpclb       | |
//  | |              |            |    ManualResolver  ClientConn   | |
//  | |              |            |     +              +            | |
//  | |              |            |     |              | Transient  | |
//  | |              |            |     |              | Failure    | |
//  | |              |            |     |  <---------  |            | |
//  | |              | <--------------- |  ResolveNow  |            | |
//  | |  <---------  | ResolveNow |     |              |            | |
//  | |  ResolveNow  |            |     |              |            | |
//  | |              |            |     |              |            | |
//  | +              +            |     +              +            | |
//  |                             +---------------------------------+ |
//  +-----------------------------------------------------------------+

// lbManualResolver is used by the ClientConn inside grpclb. It's a manual
// resolver with a special ResolveNow() function.
//
// When ResolveNow() is called, it calls ResolveNow() on the parent ClientConn,
// so when grpclb client lose contact with remote balancers, the parent
// ClientConn's resolver will re-resolve.
type lbManualResolver struct {
	scheme string
	ccr    resolver.ClientConn

	ccb balancer.ClientConn
}

func (r *lbManualResolver) Build(_ resolver.Target, cc resolver.ClientConn, _ resolver.BuildOption) (resolver.Resolver, error) {
	r.ccr = cc
	return r, nil
}

func (r *lbManualResolver) Scheme() string {
	return r.scheme
}

// ResolveNow calls resolveNow on the parent ClientConn.
func (r *lbManualResolver) ResolveNow(o resolver.ResolveNowOption) {
	r.ccb.ResolveNow(o)
}

// Close is a noop for Resolver.
func (*lbManualResolver) Close() {}

// NewAddress calls cc.NewAddress.
func (r *lbManualResolver) NewAddress(addrs []resolver.Address) {
	r.ccr.NewAddress(addrs)
}

// NewServiceConfig calls cc.NewServiceConfig.
func (r *lbManualResolver) NewServiceConfig(sc string) {
	r.ccr.NewServiceConfig(sc)
}
