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

package pubsub

import (
	"fmt"
	"math"
	"runtime"
	"time"

	gax "github.com/googleapis/gax-go"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
	"google.golang.org/api/transport"
	pubsubpb "google.golang.org/genproto/googleapis/pubsub/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
)

var (
	publisherProjectPathTemplate = gax.MustCompilePathTemplate("projects/{project}")
	publisherTopicPathTemplate   = gax.MustCompilePathTemplate("projects/{project}/topics/{topic}")
)

// PublisherCallOptions contains the retry settings for each method of this client.
type PublisherCallOptions struct {
	CreateTopic            []gax.CallOption
	Publish                []gax.CallOption
	GetTopic               []gax.CallOption
	ListTopics             []gax.CallOption
	ListTopicSubscriptions []gax.CallOption
	DeleteTopic            []gax.CallOption
}

func defaultPublisherClientOptions() []option.ClientOption {
	return []option.ClientOption{
		option.WithEndpoint("pubsub.googleapis.com:443"),
		option.WithScopes(
			"https://www.googleapis.com/auth/cloud-platform",
			"https://www.googleapis.com/auth/pubsub",
		),
	}
}

func defaultPublisherCallOptions() *PublisherCallOptions {
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
		{"messaging", "one_plus_delivery"}: {
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

	return &PublisherCallOptions{
		CreateTopic:            retry[[2]string{"default", "idempotent"}],
		Publish:                retry[[2]string{"messaging", "one_plus_delivery"}],
		GetTopic:               retry[[2]string{"default", "idempotent"}],
		ListTopics:             retry[[2]string{"default", "idempotent"}],
		ListTopicSubscriptions: retry[[2]string{"default", "idempotent"}],
		DeleteTopic:            retry[[2]string{"default", "idempotent"}],
	}
}

// PublisherClient is a client for interacting with Publisher.
type PublisherClient struct {
	// The connection to the service.
	conn *grpc.ClientConn

	// The gRPC API client.
	client pubsubpb.PublisherClient

	// The call options for this service.
	CallOptions *PublisherCallOptions

	// The metadata to be sent with each request.
	metadata map[string][]string
}

// NewPublisherClient creates a new publisher service client.
//
// The service that an application uses to manipulate topics, and to send
// messages to a topic.
func NewPublisherClient(ctx context.Context, opts ...option.ClientOption) (*PublisherClient, error) {
	conn, err := transport.DialGRPC(ctx, append(defaultPublisherClientOptions(), opts...)...)
	if err != nil {
		return nil, err
	}
	c := &PublisherClient{
		conn:        conn,
		client:      pubsubpb.NewPublisherClient(conn),
		CallOptions: defaultPublisherCallOptions(),
	}
	c.SetGoogleClientInfo("gax", gax.Version)
	return c, nil
}

// Connection returns the client's connection to the API service.
func (c *PublisherClient) Connection() *grpc.ClientConn {
	return c.conn
}

// Close closes the connection to the API service. The user should invoke this when
// the client is no longer required.
func (c *PublisherClient) Close() error {
	return c.conn.Close()
}

// SetGoogleClientInfo sets the name and version of the application in
// the `x-goog-api-client` header passed on each request. Intended for
// use by Google-written clients.
func (c *PublisherClient) SetGoogleClientInfo(name, version string) {
	c.metadata = map[string][]string{
		"x-goog-api-client": {fmt.Sprintf("%s/%s %s gax/%s go/%s", name, version, gapicNameVersion, gax.Version, runtime.Version())},
	}
}

// ProjectPath returns the path for the project resource.
func PublisherProjectPath(project string) string {
	path, err := publisherProjectPathTemplate.Render(map[string]string{
		"project": project,
	})
	if err != nil {
		panic(err)
	}
	return path
}

// TopicPath returns the path for the topic resource.
func PublisherTopicPath(project string, topic string) string {
	path, err := publisherTopicPathTemplate.Render(map[string]string{
		"project": project,
		"topic":   topic,
	})
	if err != nil {
		panic(err)
	}
	return path
}

// CreateTopic creates the given topic with the given name.
func (c *PublisherClient) CreateTopic(ctx context.Context, req *pubsubpb.Topic) (*pubsubpb.Topic, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *pubsubpb.Topic
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.CreateTopic(ctx, req)
		return err
	}, c.CallOptions.CreateTopic...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// Publish adds one or more messages to the topic. Returns `NOT_FOUND` if the topic
// does not exist. The message payload must not be empty; it must contain
//  either a non-empty data field, or at least one attribute.
func (c *PublisherClient) Publish(ctx context.Context, req *pubsubpb.PublishRequest) (*pubsubpb.PublishResponse, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *pubsubpb.PublishResponse
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.Publish(ctx, req)
		return err
	}, c.CallOptions.Publish...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// GetTopic gets the configuration of a topic.
func (c *PublisherClient) GetTopic(ctx context.Context, req *pubsubpb.GetTopicRequest) (*pubsubpb.Topic, error) {
	ctx = metadata.NewContext(ctx, c.metadata)
	var resp *pubsubpb.Topic
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		resp, err = c.client.GetTopic(ctx, req)
		return err
	}, c.CallOptions.GetTopic...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// ListTopics lists matching topics.
func (c *PublisherClient) ListTopics(ctx context.Context, req *pubsubpb.ListTopicsRequest) *TopicIterator {
	ctx = metadata.NewContext(ctx, c.metadata)
	it := &TopicIterator{}
	it.apiCall = func() error {
		var resp *pubsubpb.ListTopicsResponse
		err := gax.Invoke(ctx, func(ctx context.Context) error {
			var err error
			req.PageToken = it.nextPageToken
			req.PageSize = it.pageSize
			resp, err = c.client.ListTopics(ctx, req)
			return err
		}, c.CallOptions.ListTopics...)
		if err != nil {
			return err
		}
		if resp.NextPageToken == "" {
			it.atLastPage = true
		}
		it.nextPageToken = resp.NextPageToken
		it.items = resp.Topics
		return nil
	}
	return it
}

// ListTopicSubscriptions lists the name of the subscriptions for this topic.
func (c *PublisherClient) ListTopicSubscriptions(ctx context.Context, req *pubsubpb.ListTopicSubscriptionsRequest) *StringIterator {
	ctx = metadata.NewContext(ctx, c.metadata)
	it := &StringIterator{}
	it.apiCall = func() error {
		var resp *pubsubpb.ListTopicSubscriptionsResponse
		err := gax.Invoke(ctx, func(ctx context.Context) error {
			var err error
			req.PageToken = it.nextPageToken
			req.PageSize = it.pageSize
			resp, err = c.client.ListTopicSubscriptions(ctx, req)
			return err
		}, c.CallOptions.ListTopicSubscriptions...)
		if err != nil {
			return err
		}
		if resp.NextPageToken == "" {
			it.atLastPage = true
		}
		it.nextPageToken = resp.NextPageToken
		it.items = resp.Subscriptions
		return nil
	}
	return it
}

// DeleteTopic deletes the topic with the given name. Returns `NOT_FOUND` if the topic
// does not exist. After a topic is deleted, a new topic may be created with
// the same name; this is an entirely new topic with none of the old
// configuration or subscriptions. Existing subscriptions to this topic are
// not deleted, but their `topic` field is set to `_deleted-topic_`.
func (c *PublisherClient) DeleteTopic(ctx context.Context, req *pubsubpb.DeleteTopicRequest) error {
	ctx = metadata.NewContext(ctx, c.metadata)
	err := gax.Invoke(ctx, func(ctx context.Context) error {
		var err error
		_, err = c.client.DeleteTopic(ctx, req)
		return err
	}, c.CallOptions.DeleteTopic...)
	return err
}

// TopicIterator manages a stream of *pubsubpb.Topic.
type TopicIterator struct {
	// The current page data.
	items         []*pubsubpb.Topic
	atLastPage    bool
	currentIndex  int
	pageSize      int32
	nextPageToken string
	apiCall       func() error
}

// NextPage returns the next page of results.
// It will return at most the number of results specified by the last call to SetPageSize.
// If SetPageSize was never called or was called with a value less than 1,
// the page size is determined by the underlying service.
//
// NextPage may return a second return value of Done along with the last page of results. After
// NextPage returns Done, all subsequent calls to NextPage will return (nil, Done).
//
// Next and NextPage should not be used with the same iterator.
func (it *TopicIterator) NextPage() ([]*pubsubpb.Topic, error) {
	if it.atLastPage {
		// We already returned Done with the last page of items. Continue to
		// return Done, but with no items.
		return nil, Done
	}
	if err := it.apiCall(); err != nil {
		return nil, err
	}
	if it.atLastPage {
		return it.items, Done
	}
	return it.items, nil
}

// Next returns the next result. Its second return value is Done if there are no more results.
// Once next returns Done, all subsequent calls will return Done.
//
// Internally, Next retrieves results in bulk. You can call SetPageSize as a performance hint to
// affect how many results are retrieved in a single RPC.
//
// SetPageToken should not be called when using Next.
//
// Next and NextPage should not be used with the same iterator.
func (it *TopicIterator) Next() (*pubsubpb.Topic, error) {
	for it.currentIndex >= len(it.items) {
		if it.atLastPage {
			return nil, Done
		}
		if err := it.apiCall(); err != nil {
			return nil, err
		}
		it.currentIndex = 0
	}
	result := it.items[it.currentIndex]
	it.currentIndex++
	return result, nil
}

// PageSize returns the page size for all subsequent calls to NextPage.
func (it *TopicIterator) PageSize() int {
	return int(it.pageSize)
}

// SetPageSize sets the page size for all subsequent calls to NextPage.
func (it *TopicIterator) SetPageSize(pageSize int) {
	if pageSize > math.MaxInt32 {
		pageSize = math.MaxInt32
	}
	it.pageSize = int32(pageSize)
}

// SetPageToken sets the page token for the next call to NextPage, to resume the iteration from
// a previous point.
func (it *TopicIterator) SetPageToken(token string) {
	it.nextPageToken = token
}

// NextPageToken returns a page token that can be used with SetPageToken to resume
// iteration from the next page. It returns the empty string if there are no more pages.
func (it *TopicIterator) NextPageToken() string {
	return it.nextPageToken
}

// StringIterator manages a stream of string.
type StringIterator struct {
	// The current page data.
	items         []string
	atLastPage    bool
	currentIndex  int
	pageSize      int32
	nextPageToken string
	apiCall       func() error
}

// NextPage returns the next page of results.
// It will return at most the number of results specified by the last call to SetPageSize.
// If SetPageSize was never called or was called with a value less than 1,
// the page size is determined by the underlying service.
//
// NextPage may return a second return value of Done along with the last page of results. After
// NextPage returns Done, all subsequent calls to NextPage will return (nil, Done).
//
// Next and NextPage should not be used with the same iterator.
func (it *StringIterator) NextPage() ([]string, error) {
	if it.atLastPage {
		// We already returned Done with the last page of items. Continue to
		// return Done, but with no items.
		return nil, Done
	}
	if err := it.apiCall(); err != nil {
		return nil, err
	}
	if it.atLastPage {
		return it.items, Done
	}
	return it.items, nil
}

// Next returns the next result. Its second return value is Done if there are no more results.
// Once next returns Done, all subsequent calls will return Done.
//
// Internally, Next retrieves results in bulk. You can call SetPageSize as a performance hint to
// affect how many results are retrieved in a single RPC.
//
// SetPageToken should not be called when using Next.
//
// Next and NextPage should not be used with the same iterator.
func (it *StringIterator) Next() (string, error) {
	for it.currentIndex >= len(it.items) {
		if it.atLastPage {
			return "", Done
		}
		if err := it.apiCall(); err != nil {
			return "", err
		}
		it.currentIndex = 0
	}
	result := it.items[it.currentIndex]
	it.currentIndex++
	return result, nil
}

// PageSize returns the page size for all subsequent calls to NextPage.
func (it *StringIterator) PageSize() int {
	return int(it.pageSize)
}

// SetPageSize sets the page size for all subsequent calls to NextPage.
func (it *StringIterator) SetPageSize(pageSize int) {
	if pageSize > math.MaxInt32 {
		pageSize = math.MaxInt32
	}
	it.pageSize = int32(pageSize)
}

// SetPageToken sets the page token for the next call to NextPage, to resume the iteration from
// a previous point.
func (it *StringIterator) SetPageToken(token string) {
	it.nextPageToken = token
}

// NextPageToken returns a page token that can be used with SetPageToken to resume
// iteration from the next page. It returns the empty string if there are no more pages.
func (it *StringIterator) NextPageToken() string {
	return it.nextPageToken
}
