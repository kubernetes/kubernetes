// Copyright 2014 Google Inc. All Rights Reserved.
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

// Package pubsub contains a Google Cloud Pub/Sub client.
//
// This package is experimental and may make backwards-incompatible changes.
//
// More information about Google Cloud Pub/Sub is available at
// https://cloud.google.com/pubsub/docs
package pubsub // import "google.golang.org/cloud/pubsub"

import (
	"fmt"
	"os"

	raw "google.golang.org/api/pubsub/v1"
	"google.golang.org/cloud"
	"google.golang.org/cloud/internal/transport"

	"golang.org/x/net/context"
)

const (
	// ScopePubSub grants permissions to view and manage Pub/Sub
	// topics and subscriptions.
	ScopePubSub = "https://www.googleapis.com/auth/pubsub"

	// ScopeCloudPlatform grants permissions to view and manage your data
	// across Google Cloud Platform services.
	ScopeCloudPlatform = "https://www.googleapis.com/auth/cloud-platform"
)

const prodAddr = "https://pubsub.googleapis.com/"
const userAgent = "gcloud-golang-pubsub/20151008"

// Client is a Google Pub/Sub client, which may be used to perform Pub/Sub operations with a project.
// It must be constructed via NewClient.
type Client struct {
	projectID string
	s         service
}

// NewClient creates a new PubSub client.
func NewClient(ctx context.Context, projectID string, opts ...cloud.ClientOption) (*Client, error) {
	o := []cloud.ClientOption{
		cloud.WithEndpoint(baseAddr()),
		cloud.WithScopes(raw.PubsubScope, raw.CloudPlatformScope),
		cloud.WithUserAgent(userAgent),
	}
	o = append(o, opts...)
	httpClient, endpoint, err := transport.NewHTTPClient(ctx, o...)
	if err != nil {
		return nil, fmt.Errorf("dialing: %v", err)
	}

	s, err := newPubSubService(httpClient, endpoint)

	c := &Client{
		projectID: projectID,
		s:         s,
	}

	return c, nil
}

func (c *Client) fullyQualifiedProjectName() string {
	return fmt.Sprintf("projects/%s", c.projectID)
}

func baseAddr() string {
	// Environment variables for gcloud emulator:
	// https://cloud.google.com/sdk/gcloud/reference/beta/emulators/pubsub/
	if host := os.Getenv("PUBSUB_EMULATOR_HOST"); host != "" {
		return "http://" + host + "/"
	}
	return prodAddr
}
