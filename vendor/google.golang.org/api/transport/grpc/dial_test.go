// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package grpc

import (
	"errors"
	"net"
	"testing"
	"time"

	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
)

// Check that user optioned grpc.WithDialer option overrides the App Engine hook.
func TestGRPCHook(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	expected := false

	appengineDialerHook = (func(ctx context.Context) grpc.DialOption {
		return grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			t.Error("did not expect a call to notExpected dialer, got one")
			cancel()
			return nil, errors.New("not expected")
		})
	})
	defer func() {
		appengineDialerHook = nil
	}()

	expectedDialer := grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
		expected = true
		cancel()
		return nil, errors.New("expected")
	})

	conn, err := Dial(ctx,
		option.WithTokenSource(oauth2.StaticTokenSource(nil)), // No creds.
		option.WithGRPCDialOption(expectedDialer),
		option.WithEndpoint("example.google.com:443"))
	if err != nil {
		t.Errorf("DialGRPC: error %v, want nil", err)
	}

	// gRPC doesn't connect before the first call.
	grpc.Invoke(ctx, "foo", nil, nil, conn)
	conn.Close()

	if !expected {
		t.Error("expected a call to expected dialer, didn't get one")
	}
}
