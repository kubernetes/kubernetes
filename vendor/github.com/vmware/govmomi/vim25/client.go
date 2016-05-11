/*
Copyright (c) 2015-2016 VMware, Inc. All Rights Reserved.

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

package vim25

import (
	"encoding/json"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

// Client is a tiny wrapper around the vim25/soap Client that stores session
// specific state (i.e. state that only needs to be retrieved once after the
// client has been created). This means the client can be reused after
// serialization without performing additional requests for initialization.
type Client struct {
	*soap.Client

	ServiceContent types.ServiceContent

	// RoundTripper is a separate field such that the client's implementation of
	// the RoundTripper interface can be wrapped by separate implementations for
	// extra functionality (for example, reauthentication on session timeout).
	RoundTripper soap.RoundTripper
}

// NewClient creates and returns a new client wirh the ServiceContent field
// filled in.
func NewClient(ctx context.Context, rt soap.RoundTripper) (*Client, error) {
	serviceContent, err := methods.GetServiceContent(ctx, rt)
	if err != nil {
		return nil, err
	}

	c := Client{
		ServiceContent: serviceContent,
		RoundTripper:   rt,
	}

	// Set client if it happens to be a soap.Client
	if sc, ok := rt.(*soap.Client); ok {
		c.Client = sc
	}

	return &c, nil
}

// RoundTrip dispatches to the RoundTripper field.
func (c *Client) RoundTrip(ctx context.Context, req, res soap.HasFault) error {
	return c.RoundTripper.RoundTrip(ctx, req, res)
}

type marshaledClient struct {
	SoapClient     *soap.Client
	ServiceContent types.ServiceContent
}

func (c *Client) MarshalJSON() ([]byte, error) {
	m := marshaledClient{
		SoapClient:     c.Client,
		ServiceContent: c.ServiceContent,
	}

	return json.Marshal(m)
}

func (c *Client) UnmarshalJSON(b []byte) error {
	var m marshaledClient

	err := json.Unmarshal(b, &m)
	if err != nil {
		return err
	}

	*c = Client{
		Client:         m.SoapClient,
		ServiceContent: m.ServiceContent,
		RoundTripper:   m.SoapClient,
	}

	return nil
}

// Valid returns whether or not the client is valid and ready for use.
// This should be called after unmarshalling the client.
func (c *Client) Valid() bool {
	if c == nil {
		return false
	}

	if c.Client == nil {
		return false
	}

	// Use arbitrary pointer field in the service content.
	// Doesn't matter which one, as long as it is populated by default.
	if c.ServiceContent.SessionManager == nil {
		return false
	}

	return true
}

// IsVC returns true if we are connected to a vCenter
func (c *Client) IsVC() bool {
	return c.ServiceContent.About.ApiType == "VirtualCenter"
}
