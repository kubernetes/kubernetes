// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// AUTO-GENERATED CODE. DO NOT EDIT.

package monitoring

import (
	"math"
	"time"

	"cloud.google.com/go/internal/version"
	"github.com/golang/protobuf/proto"
	gax "github.com/googleapis/gax-go"
	"golang.org/x/net/context"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	"google.golang.org/api/transport"
	monitoredrespb "google.golang.org/genproto/googleapis/api/monitoredres"
	monitoringpb "google.golang.org/genproto/googleapis/monitoring/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
)

// GroupCallOptions contains the retry settings for each method of GroupClient.
type GroupCallOptions struct {
	ListGroups       []gax.CallOption
	GetGroup         []gax.CallOption
	CreateGroup      []gax.CallOption
	UpdateGroup      []gax.CallOption
	DeleteGroup      []gax.CallOption
	ListGroupMembers []gax.CallOption
}

func defaultGroupClientOptions() []option.ClientOption {
	return []option.ClientOption{
		option.WithEndpoint("monitoring.googleapis.com:443"),
		option.WithScopes(DefaultAuthScopes()...),
	}
}

func defaultGroupCallOptions() *GroupCallOptions {
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
	return &GroupCallOptions{
		ListGroups:       retry[[2]string{"default", "idempotent"}],
		GetGroup:         retry[[2]string{"default", "idempotent"}],
		CreateGroup:      retry[[2]string{"default", "non_idempotent"}],
		UpdateGroup:      retry[[2]string{"default", "idempotent"}],
		DeleteGroup:      retry[[2]string{"default", "idempotent"}],
		ListGroupMembers: retry[[2]string{"default", "idempotent"}],
	}
}

// GroupClient is a client for interacting with Stackdriver Monitoring API.
//
// Methods, except Close, may be called concurrently. However, fields must not be modified concurrently with method calls.
type GroupClient struct {
	// The connection to the service.
	conn *grpc.ClientConn

	// The gRPC API client.
	groupClient monitoringpb.GroupServiceClient

	// The call options for this service.
	CallOptions *GroupCallOptions

	// The x-goog-* metadata to be sent with each request.
	xGoogMetadata metadata.MD
}

// NewGroupClient creates a new group service client.
//
// The Group API lets you inspect and manage your
// groups (at #google.monitoring.v3.Group).
//
// A group is a named filter that is used to identify
// a collection of monitored resources. Groups are typically used to
// mirror the physical and/or logical topology of the environment.
// Because group membership is computed dynamically, monitored
// resources that are started in the future are automatically placed
// in matching groups. By using a group to name monitored resources in,
// for example, an alert policy, the target of that alert policy is
// updated automatically as monitored resources are added and removed
// from the infrastructure.
func NewGroupClient(ctx context.Context, opts ...option.ClientOption) (*GroupClient, error) {
	conn, err := transport.DialGRPC(ctx, append(defaultGroupClientOptions(), opts...)...)
	if err != nil {
		return nil, err
	}
	c := &GroupClient{
		conn:        conn,
		CallOptions: defaultGroupCallOptions(),

		groupClient: monitoringpb.NewGroupServiceClient(conn),
	}
	c.setGoogleClientInfo()
	return c, nil
}

// Connection returns the client's connection to the API service.
func (c *GroupClient) Connection() *grpc.ClientConn {
	return c.conn
}

// Close closes the connection to the API service. The user should invoke this when
// the client is no longer required.
func (c *GroupClient) Close() error {
	return c.conn.Close()
}

// setGoogleClientInfo sets the name and version of the application in
// the `x-goog-api-client` header passed on each request. Intended for
// use by Google-written clients.
func (c *GroupClient) setGoogleClientInfo(keyval ...string) {
	kv := append([]string{"gl-go", version.Go()}, keyval...)
	kv = append(kv, "gapic", version.Repo, "gax", gax.Version, "grpc", grpc.Version)
	c.xGoogMetadata = metadata.Pairs("x-goog-api-client", gax.XGoogHeader(kv...))
}

// ListGroups lists the existing groups.
func (c *GroupClient) ListGroups(ctx context.Context, req *monitoringpb.ListGroupsRequest, opts ...gax.CallOption) *GroupIterator {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.ListGroups[0:len(c.CallOptions.ListGroups):len(c.CallOptions.ListGroups)], opts...)
	it := &GroupIterator{}
	req = proto.Clone(req).(*monitoringpb.ListGroupsRequest)
	it.InternalFetch = func(pageSize int, pageToken string) ([]*monitoringpb.Group, string, error) {
		var resp *monitoringpb.ListGroupsResponse
		req.PageToken = pageToken
		if pageSize > math.MaxInt32 {
			req.PageSize = math.MaxInt32
		} else {
			req.PageSize = int32(pageSize)
		}
		err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
			var err error
			resp, err = c.groupClient.ListGroups(ctx, req, settings.GRPC...)
			return err
		}, opts...)
		if err != nil {
			return nil, "", err
		}
		return resp.Group, resp.NextPageToken, nil
	}
	fetch := func(pageSize int, pageToken string) (string, error) {
		items, nextPageToken, err := it.InternalFetch(pageSize, pageToken)
		if err != nil {
			return "", err
		}
		it.items = append(it.items, items...)
		return nextPageToken, nil
	}
	it.pageInfo, it.nextFunc = iterator.NewPageInfo(fetch, it.bufLen, it.takeBuf)
	it.pageInfo.MaxSize = int(req.PageSize)
	return it
}

// GetGroup gets a single group.
func (c *GroupClient) GetGroup(ctx context.Context, req *monitoringpb.GetGroupRequest, opts ...gax.CallOption) (*monitoringpb.Group, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.GetGroup[0:len(c.CallOptions.GetGroup):len(c.CallOptions.GetGroup)], opts...)
	var resp *monitoringpb.Group
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.groupClient.GetGroup(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// CreateGroup creates a new group.
func (c *GroupClient) CreateGroup(ctx context.Context, req *monitoringpb.CreateGroupRequest, opts ...gax.CallOption) (*monitoringpb.Group, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.CreateGroup[0:len(c.CallOptions.CreateGroup):len(c.CallOptions.CreateGroup)], opts...)
	var resp *monitoringpb.Group
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.groupClient.CreateGroup(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// UpdateGroup updates an existing group.
// You can change any group attributes except name.
func (c *GroupClient) UpdateGroup(ctx context.Context, req *monitoringpb.UpdateGroupRequest, opts ...gax.CallOption) (*monitoringpb.Group, error) {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.UpdateGroup[0:len(c.CallOptions.UpdateGroup):len(c.CallOptions.UpdateGroup)], opts...)
	var resp *monitoringpb.Group
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		resp, err = c.groupClient.UpdateGroup(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// DeleteGroup deletes an existing group.
func (c *GroupClient) DeleteGroup(ctx context.Context, req *monitoringpb.DeleteGroupRequest, opts ...gax.CallOption) error {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.DeleteGroup[0:len(c.CallOptions.DeleteGroup):len(c.CallOptions.DeleteGroup)], opts...)
	err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
		var err error
		_, err = c.groupClient.DeleteGroup(ctx, req, settings.GRPC...)
		return err
	}, opts...)
	return err
}

// ListGroupMembers lists the monitored resources that are members of a group.
func (c *GroupClient) ListGroupMembers(ctx context.Context, req *monitoringpb.ListGroupMembersRequest, opts ...gax.CallOption) *MonitoredResourceIterator {
	ctx = insertMetadata(ctx, c.xGoogMetadata)
	opts = append(c.CallOptions.ListGroupMembers[0:len(c.CallOptions.ListGroupMembers):len(c.CallOptions.ListGroupMembers)], opts...)
	it := &MonitoredResourceIterator{}
	req = proto.Clone(req).(*monitoringpb.ListGroupMembersRequest)
	it.InternalFetch = func(pageSize int, pageToken string) ([]*monitoredrespb.MonitoredResource, string, error) {
		var resp *monitoringpb.ListGroupMembersResponse
		req.PageToken = pageToken
		if pageSize > math.MaxInt32 {
			req.PageSize = math.MaxInt32
		} else {
			req.PageSize = int32(pageSize)
		}
		err := gax.Invoke(ctx, func(ctx context.Context, settings gax.CallSettings) error {
			var err error
			resp, err = c.groupClient.ListGroupMembers(ctx, req, settings.GRPC...)
			return err
		}, opts...)
		if err != nil {
			return nil, "", err
		}
		return resp.Members, resp.NextPageToken, nil
	}
	fetch := func(pageSize int, pageToken string) (string, error) {
		items, nextPageToken, err := it.InternalFetch(pageSize, pageToken)
		if err != nil {
			return "", err
		}
		it.items = append(it.items, items...)
		return nextPageToken, nil
	}
	it.pageInfo, it.nextFunc = iterator.NewPageInfo(fetch, it.bufLen, it.takeBuf)
	it.pageInfo.MaxSize = int(req.PageSize)
	return it
}

// GroupIterator manages a stream of *monitoringpb.Group.
type GroupIterator struct {
	items    []*monitoringpb.Group
	pageInfo *iterator.PageInfo
	nextFunc func() error

	// InternalFetch is for use by the Google Cloud Libraries only.
	// It is not part of the stable interface of this package.
	//
	// InternalFetch returns results from a single call to the underlying RPC.
	// The number of results is no greater than pageSize.
	// If there are no more results, nextPageToken is empty and err is nil.
	InternalFetch func(pageSize int, pageToken string) (results []*monitoringpb.Group, nextPageToken string, err error)
}

// PageInfo supports pagination. See the google.golang.org/api/iterator package for details.
func (it *GroupIterator) PageInfo() *iterator.PageInfo {
	return it.pageInfo
}

// Next returns the next result. Its second return value is iterator.Done if there are no more
// results. Once Next returns Done, all subsequent calls will return Done.
func (it *GroupIterator) Next() (*monitoringpb.Group, error) {
	var item *monitoringpb.Group
	if err := it.nextFunc(); err != nil {
		return item, err
	}
	item = it.items[0]
	it.items = it.items[1:]
	return item, nil
}

func (it *GroupIterator) bufLen() int {
	return len(it.items)
}

func (it *GroupIterator) takeBuf() interface{} {
	b := it.items
	it.items = nil
	return b
}

// MonitoredResourceIterator manages a stream of *monitoredrespb.MonitoredResource.
type MonitoredResourceIterator struct {
	items    []*monitoredrespb.MonitoredResource
	pageInfo *iterator.PageInfo
	nextFunc func() error

	// InternalFetch is for use by the Google Cloud Libraries only.
	// It is not part of the stable interface of this package.
	//
	// InternalFetch returns results from a single call to the underlying RPC.
	// The number of results is no greater than pageSize.
	// If there are no more results, nextPageToken is empty and err is nil.
	InternalFetch func(pageSize int, pageToken string) (results []*monitoredrespb.MonitoredResource, nextPageToken string, err error)
}

// PageInfo supports pagination. See the google.golang.org/api/iterator package for details.
func (it *MonitoredResourceIterator) PageInfo() *iterator.PageInfo {
	return it.pageInfo
}

// Next returns the next result. Its second return value is iterator.Done if there are no more
// results. Once Next returns Done, all subsequent calls will return Done.
func (it *MonitoredResourceIterator) Next() (*monitoredrespb.MonitoredResource, error) {
	var item *monitoredrespb.MonitoredResource
	if err := it.nextFunc(); err != nil {
		return item, err
	}
	item = it.items[0]
	it.items = it.items[1:]
	return item, nil
}

func (it *MonitoredResourceIterator) bufLen() int {
	return len(it.items)
}

func (it *MonitoredResourceIterator) takeBuf() interface{} {
	b := it.items
	it.items = nil
	return b
}
