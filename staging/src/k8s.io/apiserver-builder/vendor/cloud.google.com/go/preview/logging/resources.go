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

package logging

import (
	vkit "cloud.google.com/go/logging/apiv2"
	gax "github.com/googleapis/gax-go"
	"golang.org/x/net/context"
	"google.golang.org/api/iterator"
	mrpb "google.golang.org/genproto/googleapis/api/monitoredres"
	logpb "google.golang.org/genproto/googleapis/logging/v2"
)

// ResourceDescriptors returns a ResourceDescriptorIterator
// for iterating over MonitoredResourceDescriptors. Requires ReadScope or AdminScope.
// See https://cloud.google.com/logging/docs/api/v2/#monitored-resources for an explanation of
// monitored resources.
// See https://cloud.google.com/logging/docs/api/v2/resource-list for a list of monitored resources.
func (c *Client) ResourceDescriptors(ctx context.Context) *ResourceDescriptorIterator {
	it := &ResourceDescriptorIterator{
		ctx:    ctx,
		client: c.lClient,
		req:    &logpb.ListMonitoredResourceDescriptorsRequest{},
	}
	it.pageInfo, it.nextFunc = iterator.NewPageInfo(
		it.fetch,
		func() int { return len(it.items) },
		func() interface{} { b := it.items; it.items = nil; return b })
	return it
}

// ResourceDescriptorIterator is an iterator over MonitoredResourceDescriptors.
type ResourceDescriptorIterator struct {
	ctx      context.Context
	client   *vkit.Client
	pageInfo *iterator.PageInfo
	nextFunc func() error
	req      *logpb.ListMonitoredResourceDescriptorsRequest
	items    []*mrpb.MonitoredResourceDescriptor
}

// Next returns the next result. Its second return value is Done if there are
// no more results. Once Next returns Done, all subsequent calls will return
// Done.
func (it *ResourceDescriptorIterator) Next() (*mrpb.MonitoredResourceDescriptor, error) {
	if err := it.nextFunc(); err != nil {
		return nil, err
	}
	item := it.items[0]
	it.items = it.items[1:]
	return item, nil
}

func (it *ResourceDescriptorIterator) fetch(pageSize int, pageToken string) (string, error) {
	// TODO(jba): Do this a nicer way if the generated code supports one.
	// TODO(jba): If the above TODO can't be done, find a way to pass metadata in the call.
	client := logpb.NewLoggingServiceV2Client(it.client.Connection())
	var res *logpb.ListMonitoredResourceDescriptorsResponse
	err := gax.Invoke(it.ctx, func(ctx context.Context) error {
		it.req.PageSize = trunc32(pageSize)
		it.req.PageToken = pageToken
		var err error
		res, err = client.ListMonitoredResourceDescriptors(ctx, it.req)
		return err
	}, it.client.CallOptions.ListMonitoredResourceDescriptors...)
	if err != nil {
		return "", err
	}
	it.items = append(it.items, res.ResourceDescriptors...)
	return res.NextPageToken, nil
}
