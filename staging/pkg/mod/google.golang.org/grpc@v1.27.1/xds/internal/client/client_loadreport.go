/*
 *
 * Copyright 2019 gRPC authors.
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

package client

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/xds/internal/balancer/lrs"
)

// ReportLoad sends the load of the given clusterName from loadStore to the
// given server. If the server is not an empty string, and is different from the
// xds server, a new ClientConn will be created.
//
// The same options used for creating the Client will be used (including
// NodeProto, and dial options if necessary).
//
// It returns a function to cancel the load reporting stream. If server is
// different from xds server, the ClientConn will also be closed.
func (c *Client) ReportLoad(server string, clusterName string, loadStore lrs.Store) func() {
	var (
		cc      *grpc.ClientConn
		closeCC bool
	)
	if server == "" || server == c.opts.Config.BalancerName {
		cc = c.cc
	} else {
		dopts := append([]grpc.DialOption{c.opts.Config.Creds}, c.opts.DialOpts...)
		ccNew, err := grpc.Dial(server, dopts...)
		if err != nil {
			// An error from a non-blocking dial indicates something serious.
			grpclog.Infof("xds: failed to dial load report server {%s}: %v", server, err)
			return func() {}
		}
		cc = ccNew
		closeCC = true
	}
	ctx, cancel := context.WithCancel(context.Background())
	go loadStore.ReportTo(ctx, c.cc, clusterName, c.opts.Config.NodeProto)
	return func() {
		cancel()
		if closeCC {
			cc.Close()
		}
	}
}
