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

package client

import (
	"errors"
	"fmt"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/xds/internal/client/bootstrap"
	"google.golang.org/grpc/xds/internal/testutils"
	"google.golang.org/grpc/xds/internal/testutils/fakeserver"

	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
)

func clientOpts(balancerName string) Options {
	return Options{
		Config: bootstrap.Config{
			BalancerName: balancerName,
			Creds:        grpc.WithInsecure(),
			NodeProto:    &corepb.Node{},
		},
		// WithTimeout is deprecated. But we are OK to call it here from the
		// test, so we clearly know that the dial failed.
		DialOpts: []grpc.DialOption{grpc.WithTimeout(5 * time.Second), grpc.WithBlock()},
	}
}

func TestNew(t *testing.T) {
	fakeServer, cleanup, err := fakeserver.StartServer()
	if err != nil {
		t.Fatalf("Failed to start fake xDS server: %v", err)
	}
	defer cleanup()

	tests := []struct {
		name    string
		opts    Options
		wantErr bool
	}{
		{name: "empty-opts", opts: Options{}, wantErr: true},
		{
			name: "empty-balancer-name",
			opts: Options{
				Config: bootstrap.Config{
					Creds:     grpc.WithInsecure(),
					NodeProto: &corepb.Node{},
				},
			},
			wantErr: true,
		},
		{
			name: "empty-dial-creds",
			opts: Options{
				Config: bootstrap.Config{
					BalancerName: "dummy",
					NodeProto:    &corepb.Node{},
				},
			},
			wantErr: true,
		},
		{
			name: "empty-node-proto",
			opts: Options{
				Config: bootstrap.Config{
					BalancerName: "dummy",
					Creds:        grpc.WithInsecure(),
				},
			},
			wantErr: true,
		},
		{
			name:    "happy-case",
			opts:    clientOpts(fakeServer.Address),
			wantErr: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			c, err := New(test.opts)
			if err == nil {
				defer c.Close()
			}
			if (err != nil) != test.wantErr {
				t.Fatalf("New(%+v) = %v, wantErr: %v", test.opts, err, test.wantErr)
			}
		})
	}
}

// TestWatchService tests the happy case of registering a watcher for
// service updates and receiving a good update.
func TestWatchService(t *testing.T) {
	fakeServer, cleanup, err := fakeserver.StartServer()
	if err != nil {
		t.Fatalf("Failed to start fake xDS server: %v", err)
	}
	defer cleanup()

	xdsClient, err := New(clientOpts(fakeServer.Address))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	defer xdsClient.Close()
	t.Log("Created an xdsClient...")

	callbackCh := testutils.NewChannel()
	cancelWatch := xdsClient.WatchService(goodLDSTarget1, func(su ServiceUpdate, err error) {
		if err != nil {
			callbackCh.Send(fmt.Errorf("xdsClient.WatchService returned error: %v", err))
			return
		}
		if su.Cluster != goodClusterName1 {
			callbackCh.Send(fmt.Errorf("got clusterName: %+v, want clusterName: %+v", su.Cluster, goodClusterName1))
			return
		}
		callbackCh.Send(nil)
	})
	defer cancelWatch()
	t.Log("Registered a watcher for service updates...")

	// Make the fakeServer send LDS response.
	if _, err := fakeServer.XDSRequestChan.Receive(); err != nil {
		t.Fatalf("Timeout expired when expecting an LDS request")
	}
	fakeServer.XDSResponseChan <- &fakeserver.Response{Resp: goodLDSResponse1}

	// Make the fakeServer send RDS response.
	if _, err := fakeServer.XDSRequestChan.Receive(); err != nil {
		t.Fatalf("Timeout expired when expecting an RDS request")
	}
	fakeServer.XDSResponseChan <- &fakeserver.Response{Resp: goodRDSResponse1}
	waitForNilErr(t, callbackCh)
}

// TestWatchServiceWithNoResponseFromServer tests the case where the
// xDS server does not respond to the requests being sent out as part of
// registering a service update watcher. The underlying v2Client will timeout
// and will send us an error.
func TestWatchServiceWithNoResponseFromServer(t *testing.T) {
	fakeServer, cleanup, err := fakeserver.StartServer()
	if err != nil {
		t.Fatalf("Failed to start fake xDS server: %v", err)
	}
	defer cleanup()

	xdsClient, err := New(clientOpts(fakeServer.Address))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	defer xdsClient.Close()
	t.Log("Created an xdsClient...")

	oldWatchExpiryTimeout := defaultWatchExpiryTimeout
	defaultWatchExpiryTimeout = 500 * time.Millisecond
	defer func() {
		defaultWatchExpiryTimeout = oldWatchExpiryTimeout
	}()

	callbackCh := testutils.NewChannel()
	cancelWatch := xdsClient.WatchService(goodLDSTarget1, func(su ServiceUpdate, err error) {
		if su.Cluster != "" {
			callbackCh.Send(fmt.Errorf("got clusterName: %+v, want empty clusterName", su.Cluster))
			return
		}
		if err == nil {
			callbackCh.Send(errors.New("xdsClient.WatchService returned error non-nil error"))
			return
		}
		callbackCh.Send(nil)
	})
	defer cancelWatch()
	t.Log("Registered a watcher for service updates...")

	// Wait for one request from the client, but send no reponses.
	if _, err := fakeServer.XDSRequestChan.Receive(); err != nil {
		t.Fatalf("Timeout expired when expecting an LDS request")
	}
	waitForNilErr(t, callbackCh)
}

// TestWatchServiceEmptyRDS tests the case where the underlying
// v2Client receives an empty RDS response.
func TestWatchServiceEmptyRDS(t *testing.T) {
	fakeServer, cleanup, err := fakeserver.StartServer()
	if err != nil {
		t.Fatalf("Failed to start fake xDS server: %v", err)
	}
	defer cleanup()

	xdsClient, err := New(clientOpts(fakeServer.Address))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	defer xdsClient.Close()
	t.Log("Created an xdsClient...")

	oldWatchExpiryTimeout := defaultWatchExpiryTimeout
	defaultWatchExpiryTimeout = 500 * time.Millisecond
	defer func() {
		defaultWatchExpiryTimeout = oldWatchExpiryTimeout
	}()

	callbackCh := testutils.NewChannel()
	cancelWatch := xdsClient.WatchService(goodLDSTarget1, func(su ServiceUpdate, err error) {
		if su.Cluster != "" {
			callbackCh.Send(fmt.Errorf("got clusterName: %+v, want empty clusterName", su.Cluster))
			return
		}
		if err == nil {
			callbackCh.Send(errors.New("xdsClient.WatchService returned error non-nil error"))
			return
		}
		callbackCh.Send(nil)
	})
	defer cancelWatch()
	t.Log("Registered a watcher for service updates...")

	// Make the fakeServer send LDS response.
	if _, err := fakeServer.XDSRequestChan.Receive(); err != nil {
		t.Fatalf("Timeout expired when expecting an LDS request")
	}
	fakeServer.XDSResponseChan <- &fakeserver.Response{Resp: goodLDSResponse1}

	// Make the fakeServer send an empty RDS response.
	if _, err := fakeServer.XDSRequestChan.Receive(); err != nil {
		t.Fatalf("Timeout expired when expecting an RDS request")
	}
	fakeServer.XDSResponseChan <- &fakeserver.Response{Resp: noVirtualHostsInRDSResponse}
	waitForNilErr(t, callbackCh)
}

// TestWatchServiceWithClientClose tests the case where xDS responses are
// received after the client is closed, and we make sure that the registered
// watcher callback is not invoked.
func TestWatchServiceWithClientClose(t *testing.T) {
	fakeServer, cleanup, err := fakeserver.StartServer()
	if err != nil {
		t.Fatalf("Failed to start fake xDS server: %v", err)
	}
	defer cleanup()

	xdsClient, err := New(clientOpts(fakeServer.Address))
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	defer xdsClient.Close()
	t.Log("Created an xdsClient...")

	callbackCh := testutils.NewChannel()
	cancelWatch := xdsClient.WatchService(goodLDSTarget1, func(su ServiceUpdate, err error) {
		callbackCh.Send(errors.New("watcher callback invoked after client close"))
	})
	defer cancelWatch()
	t.Log("Registered a watcher for service updates...")

	// Make the fakeServer send LDS response.
	if _, err := fakeServer.XDSRequestChan.Receive(); err != nil {
		t.Fatalf("Timeout expired when expecting an LDS request")
	}
	fakeServer.XDSResponseChan <- &fakeserver.Response{Resp: goodLDSResponse1}

	xdsClient.Close()
	t.Log("Closing the xdsClient...")

	// Push an RDS response from the fakeserver
	fakeServer.XDSResponseChan <- &fakeserver.Response{Resp: goodRDSResponse1}
	if cbErr, err := callbackCh.Receive(); err != testutils.ErrRecvTimeout {
		t.Fatal(cbErr)
	}

}
