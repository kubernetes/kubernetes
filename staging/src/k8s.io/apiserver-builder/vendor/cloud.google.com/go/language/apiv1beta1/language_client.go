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

package language

import (
	"fmt"
	"runtime"
	"time"

	gax "github.com/googleapis/gax-go"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
	"google.golang.org/api/transport"
	languagepb "google.golang.org/genproto/googleapis/cloud/language/v1beta1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
)

// CallOptions contains the retry settings for each method of this client.
type CallOptions struct {
	AnalyzeSentiment []gax.CallOption
	AnalyzeEntities  []gax.CallOption
	AnnotateText     []gax.CallOption
}

func defaultClientOptions() []option.ClientOption {
	return []option.ClientOption{
		option.WithEndpoint("language.googleapis.com:443"),
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
		AnalyzeSentiment: retry[[2]string{"default", "idempotent"}],
		AnalyzeEntities:  retry[[2]string{"default", "idempotent"}],
		AnnotateText:     retry[[2]string{"default", "idempotent"}],
	}
}

// Client is a client for interacting with LanguageService.
type Client struct {
	// The connection to the service.
	conn *grpc.ClientConn

	// The gRPC API client.
	client languagepb.LanguageServiceClient

	// The call options for this service.
	CallOptions *CallOptions

	// The metadata to be sent with each request.
	metadata map[string][]string
}

// NewClient creates a new language service client.
//
// Provides text analysis operations such as sentiment analysis and entity
// recognition.
func NewClient(ctx context.Context, opts ...option.ClientOption) (*Client, error) {
	conn, err := transport.DialGRPC(ctx, append(defaultClientOptions(), opts...)...)
	if err != nil {
		return nil, err
	}
	c := &Client{
		conn:        conn,
		client:      languagepb.NewLanguageServiceClient(conn),
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

// AnalyzeSentiment analyzes the sentiment of the provided text.
func (c *Client) AnalyzeSentiment(ctx context.Context, req *languagepb.AnalyzeSentimentRequest) (*languagepb.AnalyzeSentimentResponse, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *languagepb.AnalyzeSentimentResponse
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.AnalyzeSentiment(ctx, req)
		return err
	}, c.CallOptions.AnalyzeSentiment...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// AnalyzeEntities finds named entities (currently finds proper names) in the text,
// entity types, salience, mentions for each entity, and other properties.
func (c *Client) AnalyzeEntities(ctx context.Context, req *languagepb.AnalyzeEntitiesRequest) (*languagepb.AnalyzeEntitiesResponse, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *languagepb.AnalyzeEntitiesResponse
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.AnalyzeEntities(ctx, req)
		return err
	}, c.CallOptions.AnalyzeEntities...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// AnnotateText advanced API that analyzes the document and provides a full set of text
// annotations, including semantic, syntactic, and sentiment information. This
// API is intended for users who are familiar with machine learning and need
// in-depth text features to build upon.
func (c *Client) AnnotateText(ctx context.Context, req *languagepb.AnnotateTextRequest) (*languagepb.AnnotateTextResponse, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *languagepb.AnnotateTextResponse
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.AnnotateText(ctx, req)
		return err
	}, c.CallOptions.AnnotateText...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}
