/*
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
 *
 */

// Package resolver implements the xds resolver, that does LDS and RDS to find
// the cluster to use.
package resolver

import (
	"context"
	"fmt"

	"google.golang.org/grpc"
	"google.golang.org/grpc/attributes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/resolver"

	xdsinternal "google.golang.org/grpc/xds/internal"
	xdsclient "google.golang.org/grpc/xds/internal/client"
	"google.golang.org/grpc/xds/internal/client/bootstrap"
)

// xDS balancer name is xds_experimental while resolver scheme is
// xds-experimental since "_" is not a valid character in the URL.
const xdsScheme = "xds-experimental"

// For overriding in unittests.
var (
	newXDSClient = func(opts xdsclient.Options) (xdsClientInterface, error) {
		return xdsclient.New(opts)
	}
	newXDSConfig = bootstrap.NewConfig
)

func init() {
	resolver.Register(&xdsResolverBuilder{})
}

type xdsResolverBuilder struct{}

// Build helps implement the resolver.Builder interface.
//
// The xds bootstrap process is performed (and a new xds client is built) every
// time an xds resolver is built.
func (b *xdsResolverBuilder) Build(t resolver.Target, cc resolver.ClientConn, rbo resolver.BuildOptions) (resolver.Resolver, error) {
	config, err := newXDSConfig()
	if err != nil {
		return nil, fmt.Errorf("xds: failed to read bootstrap file: %v", err)
	}
	if config.Creds == nil {
		// TODO: Once we start supporting a mechanism to register credential
		// types, a failure to find the credential type mentioned in the
		// bootstrap file should result in a failure, and not in using
		// credentials from the parent channel (passed through the
		// resolver.BuildOptions).
		config.Creds = defaultDialCreds(config.BalancerName, rbo)
	}

	var dopts []grpc.DialOption
	if rbo.Dialer != nil {
		dopts = []grpc.DialOption{grpc.WithContextDialer(rbo.Dialer)}
	}

	client, err := newXDSClient(xdsclient.Options{Config: *config, DialOpts: dopts})
	if err != nil {
		return nil, err
	}
	r := &xdsResolver{
		target:   t,
		cc:       cc,
		client:   client,
		updateCh: make(chan suWithError, 1),
	}
	r.ctx, r.cancelCtx = context.WithCancel(context.Background())
	r.cancelWatch = r.client.WatchService(r.target.Endpoint, r.handleServiceUpdate)

	go r.run()
	return r, nil
}

// defaultDialCreds builds a DialOption containing the credentials to be used
// while talking to the xDS server (this is done only if the xds bootstrap
// process does not return any credentials to use). If the parent channel
// contains DialCreds, we use it as is. If it contains a CredsBundle, we use
// just the transport credentials from the bundle. If we don't find any
// credentials on the parent channel, we resort to using an insecure channel.
func defaultDialCreds(balancerName string, rbo resolver.BuildOptions) grpc.DialOption {
	switch {
	case rbo.DialCreds != nil:
		if err := rbo.DialCreds.OverrideServerName(balancerName); err != nil {
			grpclog.Warningf("xds: failed to override server name in credentials: %v, using Insecure", err)
			return grpc.WithInsecure()
		}
		return grpc.WithTransportCredentials(rbo.DialCreds)
	case rbo.CredsBundle != nil:
		return grpc.WithTransportCredentials(rbo.CredsBundle.TransportCredentials())
	default:
		grpclog.Warning("xds: no credentials available, using Insecure")
		return grpc.WithInsecure()
	}
}

// Name helps implement the resolver.Builder interface.
func (*xdsResolverBuilder) Scheme() string {
	return xdsScheme
}

// xdsClientInterface contains methods from xdsClient.Client which are used by
// the resolver. This will be faked out in unittests.
type xdsClientInterface interface {
	WatchService(string, func(xdsclient.ServiceUpdate, error)) func()
	Close()
}

// suWithError wraps the ServiceUpdate and error received through a watch API
// callback, so that it can pushed onto the update channel as a single entity.
type suWithError struct {
	su  xdsclient.ServiceUpdate
	err error
}

// xdsResolver implements the resolver.Resolver interface.
//
// It registers a watcher for ServiceConfig updates with the xdsClient object
// (which performs LDS/RDS queries for the same), and passes the received
// updates to the ClientConn.
type xdsResolver struct {
	ctx       context.Context
	cancelCtx context.CancelFunc
	target    resolver.Target
	cc        resolver.ClientConn

	// The underlying xdsClient which performs all xDS requests and responses.
	client xdsClientInterface
	// A channel for the watch API callback to write service updates on to. The
	// updates are read by the run goroutine and passed on to the ClientConn.
	updateCh chan suWithError
	// cancelWatch is the function to cancel the watcher.
	cancelWatch func()
}

const jsonFormatSC = `{
    "loadBalancingConfig":[
      {
        "experimental_cds":{
          "Cluster": "%s"
        }
      }
    ]
  }`

// run is a long running goroutine which blocks on receiving service updates
// and passes it on the ClientConn.
func (r *xdsResolver) run() {
	for {
		select {
		case <-r.ctx.Done():
		case update := <-r.updateCh:
			if update.err != nil {
				r.cc.ReportError(update.err)
				return
			}
			sc := fmt.Sprintf(jsonFormatSC, update.su.Cluster)
			r.cc.UpdateState(resolver.State{
				ServiceConfig: r.cc.ParseServiceConfig(sc),
				Attributes:    attributes.New(xdsinternal.XDSClientID, r.client),
			})
		}
	}
}

// handleServiceUpdate is the callback which handles service updates. It writes
// the received update to the update channel, which is picked by the run
// goroutine.
func (r *xdsResolver) handleServiceUpdate(su xdsclient.ServiceUpdate, err error) {
	if r.ctx.Err() != nil {
		// Do not pass updates to the ClientConn once the resolver is closed.
		return
	}
	r.updateCh <- suWithError{su, err}
}

// ResolveNow is a no-op at this point.
func (*xdsResolver) ResolveNow(o resolver.ResolveNowOptions) {}

// Close closes the resolver, and also closes the underlying xdsClient.
func (r *xdsResolver) Close() {
	r.cancelWatch()
	r.client.Close()
	r.cancelCtx()
}
