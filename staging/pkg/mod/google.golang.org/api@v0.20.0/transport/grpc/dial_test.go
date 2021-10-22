// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package grpc

import (
	"context"
	"errors"
	"net"
	"os"
	"testing"
	"time"

	"golang.org/x/oauth2"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
)

// Check that user optioned grpc.WithDialer option overwrites App Engine dialer
func TestGRPCHook(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	expected := false

	appengineDialerHook = (func(ctx context.Context) grpc.DialOption {
		return grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			t.Error("did not expect a call to appengine dialer, got one")
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
		option.WithGRPCDialOption(grpc.WithBlock()))
	if err != context.Canceled {
		t.Errorf("got %v, want %v", err, context.Canceled)
	}
	if conn != nil {
		conn.Close()
		t.Error("got valid conn, want nil")
	}
	if !expected {
		t.Error("expected a call to expected dialer, didn't get one")
	}
}

func TestIsDirectPathEnabled(t *testing.T) {
	for _, testcase := range []struct {
		name     string
		endpoint string
		envVar   string
		want     bool
	}{
		{
			name:     "matches",
			endpoint: "some-api",
			envVar:   "some-api",
			want:     true,
		},
		{
			name:     "does not match",
			endpoint: "some-api",
			envVar:   "some-other-api",
			want:     false,
		},
		{
			name:     "matches in list",
			endpoint: "some-api-2",
			envVar:   "some-api-1,some-api-2,some-api-3",
			want:     true,
		},
		{
			name:     "empty env var",
			endpoint: "",
			envVar:   "",
			want:     false,
		},
		{
			name:     "trailing comma",
			endpoint: "",
			envVar:   "foo,bar,",
			want:     false,
		},
		{
			name:     "dns schemes are allowed",
			endpoint: "dns:///foo",
			envVar:   "dns:///foo",
			want:     true,
		},
		{
			name:     "non-dns schemes are disallowed",
			endpoint: "https://foo",
			envVar:   "https://foo",
			want:     false,
		},
	} {
		t.Run(testcase.name, func(t *testing.T) {
			if err := os.Setenv("GOOGLE_CLOUD_ENABLE_DIRECT_PATH", testcase.envVar); err != nil {
				t.Fatal(err)
			}

			if got := isDirectPathEnabled(testcase.endpoint); got != testcase.want {
				t.Fatalf("got %v, want %v", got, testcase.want)
			}
		})
	}
}
