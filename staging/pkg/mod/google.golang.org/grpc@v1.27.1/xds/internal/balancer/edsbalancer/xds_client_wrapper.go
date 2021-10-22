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
 *
 */

package edsbalancer

import (
	"google.golang.org/grpc"
	"google.golang.org/grpc/attributes"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/grpclog"
	xdsinternal "google.golang.org/grpc/xds/internal"
	"google.golang.org/grpc/xds/internal/balancer/lrs"
	xdsclient "google.golang.org/grpc/xds/internal/client"
	"google.golang.org/grpc/xds/internal/client/bootstrap"
)

// xdsClientInterface contains only the xds_client methods needed by EDS
// balancer. It's defined so we can override xdsclientNew function in tests.
type xdsClientInterface interface {
	WatchEDS(clusterName string, edsCb func(*xdsclient.EDSUpdate, error)) (cancel func())
	ReportLoad(server string, clusterName string, loadStore lrs.Store) (cancel func())
	Close()
}

var (
	xdsclientNew = func(opts xdsclient.Options) (xdsClientInterface, error) {
		return xdsclient.New(opts)
	}
	bootstrapConfigNew = bootstrap.NewConfig
)

// xdsclientWrapper is responsible for getting the xds client from attributes or
// creating a new xds client, and start watching EDS. The given callbacks will
// be called with EDS updates or errors.
type xdsclientWrapper struct {
	newEDSUpdate func(*xdsclient.EDSUpdate) error
	loseContact  func()
	bbo          balancer.BuildOptions
	loadStore    lrs.Store

	balancerName string
	// xdsclient could come from attributes, or created with balancerName.
	xdsclient xdsClientInterface

	// edsServiceName is the edsServiceName currently being watched, not
	// necessary the edsServiceName from service config.
	//
	// If edsServiceName from service config is an empty, this will be user's
	// dial target (because that's what we use to watch EDS).
	//
	// TODO: remove the empty string related behavior, when we switch to always
	// do CDS.
	edsServiceName   string
	cancelEDSWatch   func()
	loadReportServer *string // LRS is disabled if loadReporterServer is nil.
	cancelLoadReport func()
}

// newXDSClientWrapper creates an empty xds_client wrapper that does nothing. It
// can accept xds_client configs, to new/switch xds_client to use.
//
// The given callbacks won't be called until the underlying xds_client is
// working and sends updates.
func newXDSClientWrapper(newEDSUpdate func(*xdsclient.EDSUpdate) error, loseContact func(), bbo balancer.BuildOptions, loadStore lrs.Store) *xdsclientWrapper {
	return &xdsclientWrapper{
		newEDSUpdate: newEDSUpdate,
		loseContact:  loseContact,
		bbo:          bbo,
		loadStore:    loadStore,
	}
}

// replaceXDSClient replaces xdsclient fields to the newClient if they are
// different. If xdsclient is replaced, the balancerName field will also be
// updated to newBalancerName.
//
// If the old xdsclient is replaced, and was created locally (not from
// attributes), it will be closed.
//
// It returns whether xdsclient is replaced.
func (c *xdsclientWrapper) replaceXDSClient(newClient xdsClientInterface, newBalancerName string) bool {
	if c.xdsclient == newClient {
		return false
	}
	oldClient := c.xdsclient
	oldBalancerName := c.balancerName
	c.xdsclient = newClient
	c.balancerName = newBalancerName
	if oldBalancerName != "" {
		// OldBalancerName!="" means if the old client was not from attributes.
		oldClient.Close()
	}
	return true
}

// updateXDSClient sets xdsclient in wrapper to the correct one based on the
// attributes and service config.
//
// If client is found in attributes, it will be used, but we also need to decide
// whether to close the old client.
// - if old client was created locally (balancerName is not ""), close it and
// replace it
// - if old client was from previous attributes, only replace it, but don't
// close it
//
// If client is not found in attributes, will need to create a new one only if
// the balancerName (from bootstrap file or from service config) changed.
// - if balancer names are the same, do nothing, and return false
// - if balancer names are different, create new one, and return true
func (c *xdsclientWrapper) updateXDSClient(config *EDSConfig, attr *attributes.Attributes) bool {
	if attr != nil {
		if clientFromAttr, _ := attr.Value(xdsinternal.XDSClientID).(xdsClientInterface); clientFromAttr != nil {
			// This will also clear balancerName, to indicate that client is
			// from attributes.
			return c.replaceXDSClient(clientFromAttr, "")
		}
	}

	clientConfig, err := bootstrapConfigNew()
	if err != nil {
		// TODO: propagate this error to ClientConn, and fail RPCs if necessary.
		clientConfig = &bootstrap.Config{BalancerName: config.BalancerName}
	}

	if c.balancerName == clientConfig.BalancerName {
		return false
	}

	if clientConfig.Creds == nil {
		// TODO: Once we start supporting a mechanism to register credential
		// types, a failure to find the credential type mentioned in the
		// bootstrap file should result in a failure, and not in using
		// credentials from the parent channel (passed through the
		// resolver.BuildOptions).
		clientConfig.Creds = defaultDialCreds(clientConfig.BalancerName, c.bbo)
	}
	var dopts []grpc.DialOption
	if dialer := c.bbo.Dialer; dialer != nil {
		dopts = []grpc.DialOption{grpc.WithContextDialer(dialer)}
	}

	newClient, err := xdsclientNew(xdsclient.Options{Config: *clientConfig, DialOpts: dopts})
	if err != nil {
		// This should never fail. xdsclientnew does a non-blocking dial, and
		// all the config passed in should be validated.
		//
		// This could leave c.xdsclient as nil if this is the first update.
		grpclog.Warningf("eds: failed to create xdsclient, error: %v", err)
		return false
	}
	return c.replaceXDSClient(newClient, clientConfig.BalancerName)
}

// startEDSWatch starts the EDS watch. Caller can call this when the xds_client
// is updated, or the edsServiceName is updated.
//
// Note that if there's already a watch in progress, it's not explicitly
// canceled. Because for each xds_client, there should be only one EDS watch in
// progress. So a new EDS watch implicitly cancels the previous one.
//
// This usually means load report needs to be restarted, but this function does
// NOT do that. Caller needs to call startLoadReport separately.
func (c *xdsclientWrapper) startEDSWatch(nameToWatch string) {
	if c.xdsclient == nil {
		grpclog.Warningf("xds: xdsclient is nil when trying to start an EDS watch. This means xdsclient wasn't passed in from the resolver, and xdsclient.New failed")
		return
	}

	c.edsServiceName = nameToWatch
	c.cancelEDSWatch = c.xdsclient.WatchEDS(c.edsServiceName, func(update *xdsclient.EDSUpdate, err error) {
		if err != nil {
			// TODO: this should trigger a call to `c.loseContact`, when the
			// error indicates "lose contact".
			return
		}
		if err := c.newEDSUpdate(update); err != nil {
			grpclog.Warningf("xds: processing new EDS update failed due to %v.", err)
		}
	})
}

// startLoadReport starts load reporting. If there's already a load reporting in
// progress, it cancels that.
//
// Caller can cal this when the loadReportServer name changes, but
// edsServiceName doesn't (so we only need to restart load reporting, not EDS
// watch).
func (c *xdsclientWrapper) startLoadReport(edsServiceNameBeingWatched string, loadReportServer *string) {
	if c.xdsclient == nil {
		grpclog.Warningf("xds: xdsclient is nil when trying to start load reporting. This means xdsclient wasn't passed in from the resolver, and xdsclient.New failed")
		return
	}
	if c.loadStore != nil {
		if c.cancelLoadReport != nil {
			c.cancelLoadReport()
		}
		c.loadReportServer = loadReportServer
		if c.loadReportServer != nil {
			c.cancelLoadReport = c.xdsclient.ReportLoad(*c.loadReportServer, edsServiceNameBeingWatched, c.loadStore)
		}
	}
}

// handleUpdate applies the service config and attributes updates to the client,
// including updating the xds_client to use, and updating the EDS name to watch.
func (c *xdsclientWrapper) handleUpdate(config *EDSConfig, attr *attributes.Attributes) {
	clientChanged := c.updateXDSClient(config, attr)

	var (
		restartWatchEDS   bool
		restartLoadReport bool
	)

	// The clusterName to watch should come from CDS response, via service
	// config. If it's an empty string, fallback user's dial target.
	nameToWatch := config.EDSServiceName
	if nameToWatch == "" {
		grpclog.Warningf("eds: cluster name to watch is an empty string. Fallback to user's dial target")
		nameToWatch = c.bbo.Target.Endpoint
	}

	// Need to restart EDS watch when one of the following happens:
	// - the xds_client is updated
	// - the xds_client didn't change, but the edsServiceName changed
	//
	// Only need to restart load reporting when:
	// - no need to restart EDS, but loadReportServer name changed
	if clientChanged || c.edsServiceName != nameToWatch {
		restartWatchEDS = true
		restartLoadReport = true
	} else if !equalStringPointers(c.loadReportServer, config.LrsLoadReportingServerName) {
		restartLoadReport = true
	}

	if restartWatchEDS {
		c.startEDSWatch(nameToWatch)
	}

	if restartLoadReport {
		c.startLoadReport(nameToWatch, config.LrsLoadReportingServerName)
	}
}

func (c *xdsclientWrapper) close() {
	if c.xdsclient != nil && c.balancerName != "" {
		// Only close xdsclient if it's not from attributes.
		c.xdsclient.Close()
	}

	if c.cancelLoadReport != nil {
		c.cancelLoadReport()
	}
	if c.cancelEDSWatch != nil {
		c.cancelEDSWatch()
	}
}

// defaultDialCreds builds a DialOption containing the credentials to be used
// while talking to the xDS server (this is done only if the xds bootstrap
// process does not return any credentials to use). If the parent channel
// contains DialCreds, we use it as is. If it contains a CredsBundle, we use
// just the transport credentials from the bundle. If we don't find any
// credentials on the parent channel, we resort to using an insecure channel.
func defaultDialCreds(balancerName string, rbo balancer.BuildOptions) grpc.DialOption {
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

// equalStringPointers returns true if
// - a and b are both nil OR
// - *a == *b (and a and b are both non-nil)
func equalStringPointers(a, b *string) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return *a == *b
}
