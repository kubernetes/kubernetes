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

// AUTO-GENERATED CODE. DO NOT EDIT.

package speech

import (
	"fmt"
	"runtime"
	"time"

	gax "github.com/googleapis/gax-go"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
	"google.golang.org/api/transport"
	speechpb "google.golang.org/genproto/googleapis/cloud/speech/v1beta1"
	longrunningpb "google.golang.org/genproto/googleapis/longrunning"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
)

// CallOptions contains the retry settings for each method of this client.
type CallOptions struct {
	SyncRecognize  []gax.CallOption
	AsyncRecognize []gax.CallOption
}

func defaultClientOptions() []option.ClientOption {
	return []option.ClientOption{
		option.WithEndpoint("speech.googleapis.com:443"),
		option.WithScopes(
			"https://www.googleapis.com/auth/cloud-platform",
		),
	}
}

func defaultCallOptions() *CallOptions {
	retry := map[[2]string][]gax.CallOption{
		{"default", "idempotent"}: {
			gax.WithRetry(func() gax.Retryer {
				return gax.OnCodes([]codes.Code{
					codes.DeadlineExceeded,
					codes.Unavailable,
				}, gax.Backoff{
					Initial:    100 * time.Millisecond,
					Max:        60000 * time.Millisecond,
					Multiplier: 1.3,
				})
			}),
		},
	}

	return &CallOptions{
		SyncRecognize:  retry[[2]string{"default", "idempotent"}],
		AsyncRecognize: retry[[2]string{"default", "idempotent"}],
	}
}

// Client is a client for interacting with Speech.
type Client struct {
	// The connection to the service.
	conn *grpc.ClientConn

	// The gRPC API client.
	client speechpb.SpeechClient

	// The call options for this service.
	CallOptions *CallOptions

	// The metadata to be sent with each request.
	metadata map[string][]string
}

// NewClient creates a new speech service client.
//
// Service that implements Google Cloud Speech API.
func NewClient(ctx context.Context, opts ...option.ClientOption) (*Client, error) {
	conn, err := transport.DialGRPC(ctx, append(defaultClientOptions(), opts...)...)
	if err != nil {
		return nil, err
	}
	c := &Client{
		conn:        conn,
		client:      speechpb.NewSpeechClient(conn),
		CallOptions: defaultCallOptions(),
	}
	c.SetGoogleClientInfo("gax", gax.Version)
	return c, nil
}

// Connection returns the client's connection to the API service.
func (c *Client) Connection() *grpc.ClientConn {
	return c.conn
}

// Close closes the connection to the API service. The user should invoke this when
// the client is no longer required.
func (c *Client) Close() error {
	return c.conn.Close()
}

// SetGoogleClientInfo sets the name and version of the application in
// the `x-goog-api-client` header passed on each request. Intended for
// use by Google-written clients.
func (c *Client) SetGoogleClientInfo(name, version string) {
	c.metadata = map[string][]string{
		"x-goog-api-client": {fmt.Sprintf("%s/%s %s gax/%s go/%s", name, version, gapicNameVersion, gax.Version, runtime.Version())},
	}
}

// SyncRecognize perform synchronous speech-recognition: receive results after all audio
// has been sent and processed.
func (c *Client) SyncRecognize(ctx context.Context, req *speechpb.SyncRecognizeRequest) (*speechpb.SyncRecognizeResponse, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *speechpb.SyncRecognizeResponse
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.SyncRecognize(ctx, req)
		return err
	}, c.CallOptions.SyncRecognize...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// AsyncRecognize perform asynchronous speech-recognition: receive results via the
// google.longrunning.Operations interface. `Operation.response` returns
// `AsyncRecognizeResponse`.
func (c *Client) AsyncRecognize(ctx context.Context, req *speechpb.AsyncRecognizeRequest) (*longrunningpb.Operation, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *longrunningpb.Operation
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.AsyncRecognize(ctx, req)
		return err
	}, c.CallOptions.AsyncRecognize...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}
