/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package portforward

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	//nolint:staticcheck // the PortForwarder API itself operates on this compatibility type
	"k8s.io/apimachinery/pkg/util/httpstream"
	streamhttp "k8s.io/streaming/pkg/httpstream"
)

// pingableConnection extends fakeConnection with a controllable Ping, like a
// SPDY transport whose peer may or may not answer.
type pingableConnection struct {
	*fakeConnection
	pingFunc func() (time.Duration, error)
}

func (c *pingableConnection) Ping() (time.Duration, error) {
	return c.pingFunc()
}

type fakeStreamingConnection struct {
	closeChan chan bool
}

func newFakeStreamingConnection() *fakeStreamingConnection {
	return &fakeStreamingConnection{closeChan: make(chan bool)}
}

func (c *fakeStreamingConnection) CreateStream(http.Header) (streamhttp.Stream, error) {
	return nil, errors.New("not supported")
}
func (c *fakeStreamingConnection) Close() error {
	if !isChanClosed(c.closeChan) {
		close(c.closeChan)
	}
	return nil
}

func (c *fakeStreamingConnection) CloseChan() <-chan bool { return c.closeChan }

func (c *fakeStreamingConnection) SetIdleTimeout(time.Duration) {}

func (c *fakeStreamingConnection) RemoveStreams(...streamhttp.Stream) {}

func isChanClosed(ch chan bool) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}

// pingableStreamingConnection adds a controllable Ping to the streaming fake.
type pingableStreamingConnection struct {
	*fakeStreamingConnection
	pingFunc func() (time.Duration, error)
}

func (c *pingableStreamingConnection) Ping() (time.Duration, error) {
	return c.pingFunc()
}

func newKeepalivePortForwarder(t *testing.T, conn httpstream.Connection, interval, timeout time.Duration) (cancel context.CancelFunc, errChan <-chan error) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	pf, err := NewOnAddressesWithOptions(ctx, &fakeDialer{conn: conn, negotiatedProtocol: PortForwardProtocolV1Name},
		[]string{"localhost"}, []string{":5000"}, make(chan struct{}), io.Discard, io.Discard,
		WithSessionKeepalive(interval, timeout))
	require.NoError(t, err)
	errs := make(chan error, 1)
	go func() {
		errs <- pf.ForwardPorts()
	}()
	select {
	case <-pf.Ready:
	case err := <-errs:
		t.Fatalf("forwarding stopped before becoming ready: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("forwarding did not become ready")
	}
	return cancel, errs
}

func requireForwardingStopped(t *testing.T, errChan <-chan error, wantErr error) {
	t.Helper()
	select {
	case err := <-errChan:
		require.ErrorIs(t, err, wantErr)
	case <-time.After(5 * time.Second):
		t.Fatal("forwarding did not stop")
	}
}

// A peer that stops answering pings looks exactly like a half-open transport:
// writes succeed, no close notification ever arrives. The monitor must end the
// session on its own.
func TestSessionKeepaliveClosesUnresponsiveTransport(t *testing.T) {
	unblock := make(chan struct{})
	defer close(unblock)
	conn := &pingableConnection{
		fakeConnection: newFakeConnection(),
		pingFunc: func() (time.Duration, error) {
			<-unblock
			return 0, errors.New("connection closed")
		},
	}

	_, errChan := newKeepalivePortForwarder(t, conn, 20*time.Millisecond, 100*time.Millisecond)
	requireForwardingStopped(t, errChan, ErrLostConnectionToPod)
	require.True(t, conn.closed, "monitor should close the unresponsive transport")
}

func TestSessionKeepaliveClosesTransportOnPingError(t *testing.T) {
	conn := &pingableConnection{
		fakeConnection: newFakeConnection(),
		pingFunc: func() (time.Duration, error) {
			return 0, errors.New("ping write failed")
		},
	}

	_, errChan := newKeepalivePortForwarder(t, conn, 20*time.Millisecond, time.Minute)
	requireForwardingStopped(t, errChan, ErrLostConnectionToPod)
	require.True(t, conn.closed, "monitor should close the failing transport")
}

func TestSessionKeepaliveHealthyTransportKeepsSession(t *testing.T) {
	var pings atomic.Int32
	conn := &pingableConnection{
		fakeConnection: newFakeConnection(),
		pingFunc: func() (time.Duration, error) {
			pings.Add(1)
			return time.Millisecond, nil
		},
	}

	_, errChan := newKeepalivePortForwarder(t, conn, 20*time.Millisecond, time.Minute)

	// Several keepalive cycles must pass without tearing the session down.
	time.Sleep(150 * time.Millisecond)
	select {
	case err := <-errChan:
		t.Fatalf("forwarding stopped under healthy pings: %v", err)
	default:
	}
	require.GreaterOrEqual(t, pings.Load(), int32(3), "expected multiple keepalive pings")

	// The regular transport-closure path still ends the session.
	_ = conn.Close()
	requireForwardingStopped(t, errChan, ErrLostConnectionToPod)
}

// Transports without ping support must not be killed by the monitor; the
// session keeps relying on transport closure alone.
func TestSessionKeepaliveSkipsTransportWithoutPingSupport(t *testing.T) {
	conn := newFakeConnection()

	stop, errChan := newKeepalivePortForwarder(t, conn, 20*time.Millisecond, 100*time.Millisecond)

	time.Sleep(150 * time.Millisecond)
	select {
	case err := <-errChan:
		t.Fatalf("forwarding stopped although monitoring should be skipped: %v", err)
	default:
	}
	require.False(t, conn.closed, "monitor must not close a transport it cannot ping")

	stop()
	requireForwardingStopped(t, errChan, nil)
}

func TestSessionKeepaliveIgnoresNonPositiveValues(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for _, tc := range []struct {
		interval, timeout time.Duration
	}{
		{0, time.Second},
		{time.Second, 0},
		{-time.Second, -time.Second},
	} {
		pf, err := NewOnAddressesWithOptions(ctx, &fakeDialer{}, []string{"localhost"}, []string{":5000"},
			make(chan struct{}), io.Discard, io.Discard, WithSessionKeepalive(tc.interval, tc.timeout))
		require.NoError(t, err)
		require.Zero(t, pf.keepaliveInterval, "interval %v should disable keepalive", tc.interval)
		require.Zero(t, pf.keepaliveTimeout, "timeout %v should disable keepalive", tc.timeout)
	}
}

// The streaming constructor wraps every connection in compatConnectionAdapter;
// its Ping must forward to pingable delegates and report unsupported
// transports instead of failing them.
func TestCompatConnectionAdapterPing(t *testing.T) {
	pingable := &pingableStreamingConnection{
		fakeStreamingConnection: newFakeStreamingConnection(),
		pingFunc: func() (time.Duration, error) {
			return 42 * time.Millisecond, nil
		},
	}
	adapter := &compatConnectionAdapter{delegate: pingable}
	latency, err := adapter.Ping()
	require.NoError(t, err)
	require.Equal(t, 42*time.Millisecond, latency)

	adapter = &compatConnectionAdapter{delegate: newFakeStreamingConnection()}
	_, err = adapter.Ping()
	require.ErrorIs(t, err, errPingUnsupported)
}
