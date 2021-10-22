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
	"errors"
	"testing"

	xdspb "github.com/envoyproxy/go-control-plane/envoy/api/v2"
	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
	"github.com/golang/protobuf/proto"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/attributes"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/resolver"
	xdsinternal "google.golang.org/grpc/xds/internal"
	xdsclient "google.golang.org/grpc/xds/internal/client"
	"google.golang.org/grpc/xds/internal/client/bootstrap"
	"google.golang.org/grpc/xds/internal/testutils"
	"google.golang.org/grpc/xds/internal/testutils/fakeclient"
	"google.golang.org/grpc/xds/internal/testutils/fakeserver"
)

const (
	edsType = "type.googleapis.com/envoy.api.v2.ClusterLoadAssignment"
)

var (
	testServiceName    = "test/foo"
	testEDSClusterName = "test/service/eds"
)

// TestClientWrapperWatchEDS verifies that the clientWrapper registers an
// EDS watch for expected resource upon receiving an update from the top-level
// edsBalancer.
//
// The test does the following:
// * Starts a fake xDS server.
// * Creates a clientWrapper.
// * Sends updates with different edsServiceNames and expects new watches to be
//   registered.
func (s) TestClientWrapperWatchEDS(t *testing.T) {
	fakeServer, cleanup, err := fakeserver.StartServer()
	if err != nil {
		t.Fatalf("Failed to start fake xDS server: %v", err)
	}
	defer cleanup()

	cw := newXDSClientWrapper(nil, nil, balancer.BuildOptions{Target: resolver.Target{Endpoint: testServiceName}}, nil)
	defer cw.close()

	for _, test := range []struct {
		name             string
		edsServiceName   string
		wantResourceName string
	}{
		{
			// Update with an empty edsServiceName should trigger an EDS watch
			// for the user's dial target.
			name:             "empty-edsServiceName",
			edsServiceName:   "",
			wantResourceName: testServiceName,
		},
		{
			// Update with an non-empty edsServiceName should trigger an EDS
			// watch for the same.
			name:             "first-non-empty-edsServiceName",
			edsServiceName:   "foobar-1",
			wantResourceName: "foobar-1",
		},
		{
			// Also test the case where the edsServerName changes from one
			// non-empty name to another, and make sure a new watch is
			// registered.
			name:             "second-non-empty-edsServiceName",
			edsServiceName:   "foobar-2",
			wantResourceName: "foobar-2",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			oldBootstrapConfigNew := bootstrapConfigNew
			bootstrapConfigNew = func() (*bootstrap.Config, error) {
				return &bootstrap.Config{
					BalancerName: fakeServer.Address,
					Creds:        grpc.WithInsecure(),
					NodeProto:    &corepb.Node{},
				}, nil
			}
			defer func() { bootstrapConfigNew = oldBootstrapConfigNew }()
			cw.handleUpdate(&EDSConfig{
				BalancerName:   fakeServer.Address,
				EDSServiceName: test.edsServiceName,
			}, nil)

			req, err := fakeServer.XDSRequestChan.Receive()
			if err != nil {
				t.Fatalf("EDS RPC failed with err: %v", err)
			}
			edsReq := req.(*fakeserver.Request)
			if edsReq.Err != nil {
				t.Fatalf("EDS RPC failed with err: %v", edsReq.Err)
			}

			wantReq := &xdspb.DiscoveryRequest{
				TypeUrl:       edsType,
				ResourceNames: []string{test.wantResourceName},
				Node:          &corepb.Node{},
			}
			if !proto.Equal(edsReq.Req, wantReq) {
				t.Fatalf("got EDS request %v, expected: %v, diff: %s", edsReq.Req, wantReq, cmp.Diff(edsReq.Req, wantReq, cmp.Comparer(proto.Equal)))
			}
		})
	}
}

// TestClientWrapperHandleUpdateError verifies that the clientWrapper handles
// errors from the edsWatch callback appropriately.
//
// The test does the following:
// * Creates a clientWrapper.
// * Creates a fakeclient.Client and passes it to the clientWrapper in attributes.
// * Verifies the clientWrapper registers an EDS watch.
// * Forces the fakeclient.Client to invoke the registered EDS watch callback with
//   an error. Verifies that the wrapper does not invoke the top-level
//   edsBalancer with the received error.
func (s) TestClientWrapperHandleUpdateError(t *testing.T) {
	edsRespChan := testutils.NewChannel()
	newEDS := func(update *xdsclient.EDSUpdate) error {
		edsRespChan.Send(update)
		return nil
	}

	cw := newXDSClientWrapper(newEDS, nil, balancer.BuildOptions{Target: resolver.Target{Endpoint: testServiceName}}, nil)
	defer cw.close()

	xdsC := fakeclient.NewClient()
	cw.handleUpdate(&EDSConfig{EDSServiceName: testEDSClusterName}, attributes.New(xdsinternal.XDSClientID, xdsC))
	gotCluster, err := xdsC.WaitForWatchEDS()
	if err != nil {
		t.Fatalf("xdsClient.WatchEDS failed with error: %v", err)
	}
	if gotCluster != testEDSClusterName {
		t.Fatalf("xdsClient.WatchEDS() called with cluster: %v, want %v", gotCluster, testEDSClusterName)
	}
	xdsC.InvokeWatchEDSCallback(nil, errors.New("EDS watch callback error"))

	// The callback is called with an error, expect no update from edsRespChan.
	//
	// TODO: check for loseContact() when errors indicating "lose contact" are
	// handled correctly.
	if gotUpdate, gotErr := edsRespChan.Receive(); gotErr != testutils.ErrRecvTimeout {
		t.Fatalf("edsBalancer got edsUpdate {%+v, %v}, when none was expected", gotUpdate, gotErr)
	}
}

// TestClientWrapperGetsXDSClientInAttributes verfies the case where the
// clientWrapper receives the xdsClient to use in the attributes section of the
// update.
func (s) TestClientWrapperGetsXDSClientInAttributes(t *testing.T) {
	oldxdsclientNew := xdsclientNew
	xdsclientNew = func(_ xdsclient.Options) (xdsClientInterface, error) {
		t.Fatalf("unexpected call to xdsclientNew when xds_client is set in attributes")
		return nil, nil
	}
	defer func() { xdsclientNew = oldxdsclientNew }()

	cw := newXDSClientWrapper(nil, nil, balancer.BuildOptions{Target: resolver.Target{Endpoint: testServiceName}}, nil)
	defer cw.close()

	// Verify that the eds watch is registered for the expected resource name.
	xdsC1 := fakeclient.NewClient()
	cw.handleUpdate(&EDSConfig{EDSServiceName: testEDSClusterName}, attributes.New(xdsinternal.XDSClientID, xdsC1))
	gotCluster, err := xdsC1.WaitForWatchEDS()
	if err != nil {
		t.Fatalf("xdsClient.WatchEDS failed with error: %v", err)
	}
	if gotCluster != testEDSClusterName {
		t.Fatalf("xdsClient.WatchEDS() called with cluster: %v, want %v", gotCluster, testEDSClusterName)
	}

	// Pass a new client in the attributes. Verify that the watch is
	// re-registered on the new client, and that the old client is not closed
	// (because clientWrapper only closes clients that it creates, it does not
	// close client that are passed through attributes).
	xdsC2 := fakeclient.NewClient()
	cw.handleUpdate(&EDSConfig{EDSServiceName: testEDSClusterName}, attributes.New(xdsinternal.XDSClientID, xdsC2))
	gotCluster, err = xdsC2.WaitForWatchEDS()
	if err != nil {
		t.Fatalf("xdsClient.WatchEDS failed with error: %v", err)
	}
	if gotCluster != testEDSClusterName {
		t.Fatalf("xdsClient.WatchEDS() called with cluster: %v, want %v", gotCluster, testEDSClusterName)
	}

	if err := xdsC1.WaitForClose(); err != testutils.ErrRecvTimeout {
		t.Fatalf("clientWrapper closed xdsClient received in attributes")
	}
}
