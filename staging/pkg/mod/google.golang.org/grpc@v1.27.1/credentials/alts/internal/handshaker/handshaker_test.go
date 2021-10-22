/*
 *
 * Copyright 2018 gRPC authors.
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

package handshaker

import (
	"bytes"
	"context"
	"testing"
	"time"

	grpc "google.golang.org/grpc"
	core "google.golang.org/grpc/credentials/alts/internal"
	altspb "google.golang.org/grpc/credentials/alts/internal/proto/grpc_gcp"
	"google.golang.org/grpc/credentials/alts/internal/testutil"
)

var (
	testRecordProtocol = rekeyRecordProtocolName
	testKey            = []byte{
		// 44 arbitrary bytes.
		0x1f, 0x8b, 0x08, 0x00, 0x00, 0x09, 0x6e, 0x88, 0x02, 0xff, 0xe2, 0xd2, 0x4c, 0xce, 0x4f, 0x49,
		0x08, 0x00, 0x00, 0x09, 0x6e, 0x88, 0x02, 0xff, 0xe2, 0xd2, 0x4c, 0xce, 0x4f, 0x49, 0x1f, 0x8b,
		0xd2, 0x4c, 0xce, 0x08, 0x00, 0x00, 0x09, 0x6e, 0x88, 0x02, 0xff, 0xe2,
	}
	testServiceAccount        = "test_service_account"
	testTargetServiceAccounts = []string{testServiceAccount}
	testClientIdentity        = &altspb.Identity{
		IdentityOneof: &altspb.Identity_Hostname{
			Hostname: "i_am_a_client",
		},
	}
)

// testRPCStream mimics a altspb.HandshakerService_DoHandshakeClient object.
type testRPCStream struct {
	grpc.ClientStream
	t        *testing.T
	isClient bool
	// The resp expected to be returned by Recv(). Make sure this is set to
	// the content the test requires before Recv() is invoked.
	recvBuf *altspb.HandshakerResp
	// false if it is the first access to Handshaker service on Envelope.
	first bool
	// useful for testing concurrent calls.
	delay time.Duration
}

func (t *testRPCStream) Recv() (*altspb.HandshakerResp, error) {
	resp := t.recvBuf
	t.recvBuf = nil
	return resp, nil
}

func (t *testRPCStream) Send(req *altspb.HandshakerReq) error {
	var resp *altspb.HandshakerResp
	if !t.first {
		// Generate the bytes to be returned by Recv() for the initial
		// handshaking.
		t.first = true
		if t.isClient {
			resp = &altspb.HandshakerResp{
				OutFrames: testutil.MakeFrame("ClientInit"),
				// Simulate consuming ServerInit.
				BytesConsumed: 14,
			}
		} else {
			resp = &altspb.HandshakerResp{
				OutFrames: testutil.MakeFrame("ServerInit"),
				// Simulate consuming ClientInit.
				BytesConsumed: 14,
			}
		}
	} else {
		// Add delay to test concurrent calls.
		cleanup := stat.Update()
		defer cleanup()
		time.Sleep(t.delay)

		// Generate the response to be returned by Recv() for the
		// follow-up handshaking.
		result := &altspb.HandshakerResult{
			RecordProtocol: testRecordProtocol,
			KeyData:        testKey,
		}
		resp = &altspb.HandshakerResp{
			Result: result,
			// Simulate consuming ClientFinished or ServerFinished.
			BytesConsumed: 18,
		}
	}
	t.recvBuf = resp
	return nil
}

func (t *testRPCStream) CloseSend() error {
	return nil
}

var stat testutil.Stats

func TestClientHandshake(t *testing.T) {
	for _, testCase := range []struct {
		delay              time.Duration
		numberOfHandshakes int
	}{
		{0 * time.Millisecond, 1},
		{100 * time.Millisecond, 10 * maxPendingHandshakes},
	} {
		errc := make(chan error)
		stat.Reset()
		for i := 0; i < testCase.numberOfHandshakes; i++ {
			stream := &testRPCStream{
				t:        t,
				isClient: true,
			}
			// Preload the inbound frames.
			f1 := testutil.MakeFrame("ServerInit")
			f2 := testutil.MakeFrame("ServerFinished")
			in := bytes.NewBuffer(f1)
			in.Write(f2)
			out := new(bytes.Buffer)
			tc := testutil.NewTestConn(in, out)
			chs := &altsHandshaker{
				stream: stream,
				conn:   tc,
				clientOpts: &ClientHandshakerOptions{
					TargetServiceAccounts: testTargetServiceAccounts,
					ClientIdentity:        testClientIdentity,
				},
				side: core.ClientSide,
			}
			go func() {
				_, context, err := chs.ClientHandshake(context.Background())
				if err == nil && context == nil {
					panic("expected non-nil ALTS context")
				}
				errc <- err
				chs.Close()
			}()
		}

		// Ensure all errors are expected.
		for i := 0; i < testCase.numberOfHandshakes; i++ {
			if err := <-errc; err != nil && err != errDropped {
				t.Errorf("ClientHandshake() = _, %v, want _, <nil> or %v", err, errDropped)
			}
		}

		// Ensure that there are no concurrent calls more than the limit.
		if stat.MaxConcurrentCalls > maxPendingHandshakes {
			t.Errorf("Observed %d concurrent handshakes; want <= %d", stat.MaxConcurrentCalls, maxPendingHandshakes)
		}
	}
}

func TestServerHandshake(t *testing.T) {
	for _, testCase := range []struct {
		delay              time.Duration
		numberOfHandshakes int
	}{
		{0 * time.Millisecond, 1},
		{100 * time.Millisecond, 10 * maxPendingHandshakes},
	} {
		errc := make(chan error)
		stat.Reset()
		for i := 0; i < testCase.numberOfHandshakes; i++ {
			stream := &testRPCStream{
				t:        t,
				isClient: false,
			}
			// Preload the inbound frames.
			f1 := testutil.MakeFrame("ClientInit")
			f2 := testutil.MakeFrame("ClientFinished")
			in := bytes.NewBuffer(f1)
			in.Write(f2)
			out := new(bytes.Buffer)
			tc := testutil.NewTestConn(in, out)
			shs := &altsHandshaker{
				stream:     stream,
				conn:       tc,
				serverOpts: DefaultServerHandshakerOptions(),
				side:       core.ServerSide,
			}
			go func() {
				_, context, err := shs.ServerHandshake(context.Background())
				if err == nil && context == nil {
					panic("expected non-nil ALTS context")
				}
				errc <- err
				shs.Close()
			}()
		}

		// Ensure all errors are expected.
		for i := 0; i < testCase.numberOfHandshakes; i++ {
			if err := <-errc; err != nil && err != errDropped {
				t.Errorf("ServerHandshake() = _, %v, want _, <nil> or %v", err, errDropped)
			}
		}

		// Ensure that there are no concurrent calls more than the limit.
		if stat.MaxConcurrentCalls > maxPendingHandshakes {
			t.Errorf("Observed %d concurrent handshakes; want <= %d", stat.MaxConcurrentCalls, maxPendingHandshakes)
		}
	}
}

// testUnresponsiveRPCStream is used for testing the PeerNotResponding case.
type testUnresponsiveRPCStream struct {
	grpc.ClientStream
}

func (t *testUnresponsiveRPCStream) Recv() (*altspb.HandshakerResp, error) {
	return &altspb.HandshakerResp{}, nil
}

func (t *testUnresponsiveRPCStream) Send(req *altspb.HandshakerReq) error {
	return nil
}

func (t *testUnresponsiveRPCStream) CloseSend() error {
	return nil
}

func TestPeerNotResponding(t *testing.T) {
	stream := &testUnresponsiveRPCStream{}
	chs := &altsHandshaker{
		stream: stream,
		conn:   testutil.NewUnresponsiveTestConn(),
		clientOpts: &ClientHandshakerOptions{
			TargetServiceAccounts: testTargetServiceAccounts,
			ClientIdentity:        testClientIdentity,
		},
		side: core.ClientSide,
	}
	_, context, err := chs.ClientHandshake(context.Background())
	chs.Close()
	if context != nil {
		t.Error("expected non-nil ALTS context")
	}
	if got, want := err, core.PeerNotRespondingError; got != want {
		t.Errorf("ClientHandshake() = %v, want %v", got, want)
	}
}
