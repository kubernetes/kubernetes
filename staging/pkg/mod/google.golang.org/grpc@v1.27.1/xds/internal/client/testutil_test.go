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
	"reflect"
	"testing"

	xdspb "github.com/envoyproxy/go-control-plane/envoy/api/v2"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/xds/internal/testutils"
	"google.golang.org/grpc/xds/internal/testutils/fakeserver"
)

type watchHandleTestcase struct {
	responseToHandle *xdspb.DiscoveryResponse
	wantHandleErr    bool
	wantUpdate       interface{}
	wantUpdateErr    bool

	// Only one of the following should be non-nil. The one corresponding with
	// typeURL will be called.
	ldsWatch      func(target string, ldsCb ldsCallback) (cancel func())
	rdsWatch      func(routeName string, rdsCb rdsCallback) (cancel func())
	cdsWatch      func(clusterName string, cdsCb cdsCallback) (cancel func())
	edsWatch      func(clusterName string, edsCb edsCallback) (cancel func())
	watchReqChan  *testutils.Channel // The request sent for watch will be sent to this channel.
	handleXDSResp func(response *xdspb.DiscoveryResponse) error
}

// testWatchHandle is called to test response handling for each xDS.
//
// It starts the xDS watch as configured in test, waits for the fake xds server
// to receive the request (so watch callback is installed), and calls
// handleXDSResp with responseToHandle (if it's set). It then compares the
// update received by watch callback with the expected results.
func testWatchHandle(t *testing.T, test *watchHandleTestcase) {
	type updateErr struct {
		u   interface{}
		err error
	}
	gotUpdateCh := testutils.NewChannel()

	var cancelWatch func()
	// Register the watcher, this will also trigger the v2Client to send the xDS
	// request.
	switch {
	case test.ldsWatch != nil:
		cancelWatch = test.ldsWatch(goodLDSTarget1, func(u ldsUpdate, err error) {
			t.Logf("in v2c.watchLDS callback, ldsUpdate: %+v, err: %v", u, err)
			gotUpdateCh.Send(updateErr{u, err})
		})
	case test.rdsWatch != nil:
		cancelWatch = test.rdsWatch(goodRouteName1, func(u rdsUpdate, err error) {
			t.Logf("in v2c.watchRDS callback, rdsUpdate: %+v, err: %v", u, err)
			gotUpdateCh.Send(updateErr{u, err})
		})
	case test.cdsWatch != nil:
		cancelWatch = test.cdsWatch(clusterName1, func(u CDSUpdate, err error) {
			t.Logf("in v2c.watchCDS callback, cdsUpdate: %+v, err: %v", u, err)
			gotUpdateCh.Send(updateErr{u, err})
		})
	case test.edsWatch != nil:
		cancelWatch = test.edsWatch(goodEDSName, func(u *EDSUpdate, err error) {
			t.Logf("in v2c.watchEDS callback, edsUpdate: %+v, err: %v", u, err)
			gotUpdateCh.Send(updateErr{*u, err})
		})
	default:
		t.Fatalf("no watch() is set")
	}
	defer cancelWatch()

	// Wait till the request makes it to the fakeServer. This ensures that
	// the watch request has been processed by the v2Client.
	if _, err := test.watchReqChan.Receive(); err != nil {
		t.Fatalf("Timeout waiting for an xDS request: %v", err)
	}

	// Directly push the response through a call to handleXDSResp. This bypasses
	// the fakeServer, so it's only testing the handle logic. Client response
	// processing is covered elsewhere.
	//
	// Also note that this won't trigger ACK, so there's no need to clear the
	// request channel afterwards.
	if err := test.handleXDSResp(test.responseToHandle); (err != nil) != test.wantHandleErr {
		t.Fatalf("v2c.handleRDSResponse() returned err: %v, wantErr: %v", err, test.wantHandleErr)
	}

	// If the test doesn't expect the callback to be invoked, verify that no
	// update or error is pushed to the callback.
	//
	// Cannot directly compare test.wantUpdate with nil (typed vs non-typed nil:
	// https://golang.org/doc/faq#nil_error).
	if c := test.wantUpdate; c == nil || (reflect.ValueOf(c).Kind() == reflect.Ptr && reflect.ValueOf(c).IsNil()) {
		update, err := gotUpdateCh.Receive()
		if err == testutils.ErrRecvTimeout {
			return
		}
		t.Fatalf("Unexpected update: +%v", update)
	}

	wantUpdate := reflect.ValueOf(test.wantUpdate).Elem().Interface()
	uErr, err := gotUpdateCh.Receive()
	if err == testutils.ErrRecvTimeout {
		t.Fatal("Timeout expecting xDS update")
	}
	gotUpdate := uErr.(updateErr).u
	opt := cmp.AllowUnexported(rdsUpdate{}, ldsUpdate{}, CDSUpdate{}, EDSUpdate{})
	if diff := cmp.Diff(gotUpdate, wantUpdate, opt); diff != "" {
		t.Fatalf("got update : %+v, want %+v, diff: %s", gotUpdate, wantUpdate, diff)
	}
	gotUpdateErr := uErr.(updateErr).err
	if (gotUpdateErr != nil) != test.wantUpdateErr {
		t.Fatalf("got xDS update error {%v}, wantErr: %v", gotUpdateErr, test.wantUpdateErr)
	}
}

// startServerAndGetCC starts a fake XDS server and also returns a ClientConn
// connected to it.
func startServerAndGetCC(t *testing.T) (*fakeserver.Server, *grpc.ClientConn, func()) {
	t.Helper()

	fs, sCleanup, err := fakeserver.StartServer()
	if err != nil {
		t.Fatalf("Failed to start fake xDS server: %v", err)
	}

	cc, ccCleanup, err := fs.XDSClientConn()
	if err != nil {
		sCleanup()
		t.Fatalf("Failed to get a clientConn to the fake xDS server: %v", err)
	}
	return fs, cc, func() {
		sCleanup()
		ccCleanup()
	}
}

// waitForNilErr waits for a nil error value to be received on the
// provided channel.
func waitForNilErr(t *testing.T, ch *testutils.Channel) {
	t.Helper()

	val, err := ch.Receive()
	if err == testutils.ErrRecvTimeout {
		t.Fatalf("Timeout expired when expecting update")
	}
	if val != nil {
		if cbErr := val.(error); cbErr != nil {
			t.Fatal(cbErr)
		}
	}
}
