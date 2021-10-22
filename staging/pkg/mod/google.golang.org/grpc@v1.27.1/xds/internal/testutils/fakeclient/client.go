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

// Package fakeclient provides a fake implementation of an xDS client.
package fakeclient

import (
	"sync"

	"google.golang.org/grpc/xds/internal/balancer/lrs"
	xdsclient "google.golang.org/grpc/xds/internal/client"
	"google.golang.org/grpc/xds/internal/testutils"
)

// Client is a fake implementation of an xds client. It exposes a bunch of
// channels to signal the occurrence of various events.
type Client struct {
	name         string
	suWatchCh    *testutils.Channel
	cdsWatchCh   *testutils.Channel
	edsWatchCh   *testutils.Channel
	suCancelCh   *testutils.Channel
	cdsCancelCh  *testutils.Channel
	edsCancelCh  *testutils.Channel
	loadReportCh *testutils.Channel
	closeCh      *testutils.Channel

	mu        sync.Mutex
	serviceCb func(xdsclient.ServiceUpdate, error)
	cdsCb     func(xdsclient.CDSUpdate, error)
	edsCb     func(*xdsclient.EDSUpdate, error)
}

// WatchService registers a LDS/RDS watch.
func (xdsC *Client) WatchService(target string, callback func(xdsclient.ServiceUpdate, error)) func() {
	xdsC.mu.Lock()
	defer xdsC.mu.Unlock()

	xdsC.serviceCb = callback
	xdsC.suWatchCh.Send(target)
	return func() {
		xdsC.suCancelCh.Send(nil)
	}
}

// WaitForWatchService waits for WatchService to be invoked on this client
// within a reasonable timeout, and returns the serviceName being watched.
func (xdsC *Client) WaitForWatchService() (string, error) {
	val, err := xdsC.suWatchCh.Receive()
	if err != nil {
		return "", err
	}
	return val.(string), err
}

// InvokeWatchServiceCallback invokes the registered service watch callback.
func (xdsC *Client) InvokeWatchServiceCallback(cluster string, err error) {
	xdsC.mu.Lock()
	defer xdsC.mu.Unlock()

	xdsC.serviceCb(xdsclient.ServiceUpdate{Cluster: cluster}, err)
}

// WatchCluster registers a CDS watch.
func (xdsC *Client) WatchCluster(clusterName string, callback func(xdsclient.CDSUpdate, error)) func() {
	xdsC.mu.Lock()
	defer xdsC.mu.Unlock()

	xdsC.cdsCb = callback
	xdsC.cdsWatchCh.Send(clusterName)
	return func() {
		xdsC.cdsCancelCh.Send(nil)
	}
}

// WaitForWatchCluster waits for WatchCluster to be invoked on this client
// within a reasonable timeout, and returns the clusterName being watched.
func (xdsC *Client) WaitForWatchCluster() (string, error) {
	val, err := xdsC.cdsWatchCh.Receive()
	if err != nil {
		return "", err
	}
	return val.(string), err
}

// InvokeWatchClusterCallback invokes the registered cdsWatch callback.
func (xdsC *Client) InvokeWatchClusterCallback(update xdsclient.CDSUpdate, err error) {
	xdsC.mu.Lock()
	defer xdsC.mu.Unlock()

	xdsC.cdsCb(update, err)
}

// WaitForCancelClusterWatch waits for a CDS watch to be cancelled within a
// reasonable timeout, and returns testutils.ErrRecvTimeout otherwise.
func (xdsC *Client) WaitForCancelClusterWatch() error {
	_, err := xdsC.cdsCancelCh.Receive()
	return err
}

// WatchEDS registers an EDS watch for provided clusterName.
func (xdsC *Client) WatchEDS(clusterName string, callback func(*xdsclient.EDSUpdate, error)) (cancel func()) {
	xdsC.mu.Lock()
	defer xdsC.mu.Unlock()

	xdsC.edsCb = callback
	xdsC.edsWatchCh.Send(clusterName)
	return func() {
		xdsC.edsCancelCh.Send(nil)
	}
}

// WaitForWatchEDS waits for WatchEDS to be invoked on this client within a
// reasonable timeout, and returns the clusterName being watched.
func (xdsC *Client) WaitForWatchEDS() (string, error) {
	val, err := xdsC.edsWatchCh.Receive()
	if err != nil {
		return "", err
	}
	return val.(string), err
}

// InvokeWatchEDSCallback invokes the registered edsWatch callback.
func (xdsC *Client) InvokeWatchEDSCallback(update *xdsclient.EDSUpdate, err error) {
	xdsC.mu.Lock()
	defer xdsC.mu.Unlock()

	xdsC.edsCb(update, err)
}

// ReportLoadArgs wraps the arguments passed to ReportLoad.
type ReportLoadArgs struct {
	// Server is the name of the server to which the load is reported.
	Server string
	// Cluster is the name of the cluster for which load is reported.
	Cluster string
}

// ReportLoad starts reporting load about clusterName to server.
func (xdsC *Client) ReportLoad(server string, clusterName string, loadStore lrs.Store) (cancel func()) {
	xdsC.loadReportCh.Send(ReportLoadArgs{Server: server, Cluster: clusterName})
	return func() {}
}

// WaitForReportLoad waits for ReportLoad to be invoked on this client within a
// reasonable timeout, and returns the arguments passed to it.
func (xdsC *Client) WaitForReportLoad() (ReportLoadArgs, error) {
	val, err := xdsC.loadReportCh.Receive()
	return val.(ReportLoadArgs), err
}

// Close closes the xds client.
func (xdsC *Client) Close() {
	xdsC.closeCh.Send(nil)
}

// WaitForClose waits for Close to be invoked on this client within a
// reasonable timeout, and returns testutils.ErrRecvTimeout otherwise.
func (xdsC *Client) WaitForClose() error {
	_, err := xdsC.closeCh.Receive()
	return err
}

// Name returns the name of the xds client.
func (xdsC *Client) Name() string {
	return xdsC.name
}

// NewClient returns a new fake xds client.
func NewClient() *Client {
	return NewClientWithName("")
}

// NewClientWithName returns a new fake xds client with the provided name. This
// is used in cases where multiple clients are created in the tests and we need
// to make sure the client is created for the expected balancer name.
func NewClientWithName(name string) *Client {
	return &Client{
		name:         name,
		suWatchCh:    testutils.NewChannel(),
		cdsWatchCh:   testutils.NewChannel(),
		edsWatchCh:   testutils.NewChannel(),
		suCancelCh:   testutils.NewChannel(),
		cdsCancelCh:  testutils.NewChannel(),
		edsCancelCh:  testutils.NewChannel(),
		loadReportCh: testutils.NewChannel(),
		closeCh:      testutils.NewChannel(),
	}
}
