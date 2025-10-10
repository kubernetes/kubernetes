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

package xdsclient

import (
	"context"
	"sync"

	"google.golang.org/grpc/internal/xds/bootstrap"
	"google.golang.org/grpc/internal/xds/clients"
	"google.golang.org/grpc/internal/xds/clients/grpctransport"
	"google.golang.org/grpc/internal/xds/clients/lrsclient"
)

// ReportLoad starts a load reporting stream to the given server. All load
// reports to the same server share the LRS stream.
//
// It returns a lrsclient.LoadStore for the user to report loads.
func (c *clientImpl) ReportLoad(server *bootstrap.ServerConfig) (*lrsclient.LoadStore, func(context.Context)) {
	load, err := c.lrsClient.ReportLoad(clients.ServerIdentifier{
		ServerURI: server.ServerURI(),
		Extensions: grpctransport.ServerIdentifierExtension{
			ConfigName: server.SelectedCreds().Type,
		},
	})
	if err != nil {
		c.logger.Warningf("Failed to create a load store to the management server to report load: %v", server, err)
		return nil, func(context.Context) {}
	}
	var loadStop sync.Once
	return load, func(ctx context.Context) {
		loadStop.Do(func() { load.Stop(ctx) })
	}
}
