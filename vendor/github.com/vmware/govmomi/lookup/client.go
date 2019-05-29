/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package lookup

import (
	"context"
	"crypto/x509"
	"encoding/base64"
	"log"
	"net/url"

	"github.com/vmware/govmomi/lookup/methods"
	"github.com/vmware/govmomi/lookup/types"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
	vim "github.com/vmware/govmomi/vim25/types"
)

const (
	Namespace = "lookup"
	Version   = "2.0"
	Path      = "/lookupservice" + vim25.Path
)

var (
	ServiceInstance = vim.ManagedObjectReference{
		Type:  "LookupServiceInstance",
		Value: "ServiceInstance",
	}
)

// Client is a soap.Client targeting the SSO Lookup Service API endpoint.
type Client struct {
	*soap.Client

	ServiceContent types.LookupServiceContent
}

// NewClient returns a client targeting the SSO Lookup Service API endpoint.
func NewClient(ctx context.Context, c *vim25.Client) (*Client, error) {
	sc := c.Client.NewServiceClient(Path, Namespace)
	sc.Version = Version

	req := types.RetrieveServiceContent{
		This: ServiceInstance,
	}

	res, err := methods.RetrieveServiceContent(ctx, sc, &req)
	if err != nil {
		return nil, err
	}

	return &Client{sc, res.Returnval}, nil
}

func (c *Client) List(ctx context.Context, filter *types.LookupServiceRegistrationFilter) ([]types.LookupServiceRegistrationInfo, error) {
	req := types.List{
		This:           *c.ServiceContent.ServiceRegistration,
		FilterCriteria: filter,
	}

	res, err := methods.List(ctx, c, &req)
	if err != nil {
		return nil, err
	}
	return res.Returnval, nil
}

func (c *Client) SiteID(ctx context.Context) (string, error) {
	req := types.GetSiteId{
		This: *c.ServiceContent.ServiceRegistration,
	}

	res, err := methods.GetSiteId(ctx, c, &req)
	if err != nil {
		return "", err
	}
	return res.Returnval, nil
}

// EndpointURL uses the Lookup Service to find the endpoint URL and thumbprint for the given filter.
// If the endpoint is found, its TLS certificate is also added to the vim25.Client's trusted host thumbprints.
// If the Lookup Service is not available, the given path is returned as the default.
func EndpointURL(ctx context.Context, c *vim25.Client, path string, filter *types.LookupServiceRegistrationFilter) string {
	if lu, err := NewClient(ctx, c); err == nil {
		info, _ := lu.List(ctx, filter)
		if len(info) != 0 && len(info[0].ServiceEndpoints) != 0 {
			endpoint := &info[0].ServiceEndpoints[0]
			path = endpoint.Url

			if u, err := url.Parse(path); err == nil {
				if c.Thumbprint(u.Host) == "" {
					c.SetThumbprint(u.Host, endpointThumbprint(endpoint))
				}
			}
		}
	}
	return path
}

// endpointThumbprint converts the base64 encoded endpoint certificate to a SHA1 thumbprint.
func endpointThumbprint(endpoint *types.LookupServiceRegistrationEndpoint) string {
	if len(endpoint.SslTrust) == 0 {
		return ""
	}
	enc := endpoint.SslTrust[0]

	b, err := base64.StdEncoding.DecodeString(enc)
	if err != nil {
		log.Printf("base64.Decode(%q): %s", enc, err)
		return ""
	}

	cert, err := x509.ParseCertificate(b)
	if err != nil {
		log.Printf("x509.ParseCertificate(%q): %s", enc, err)
		return ""
	}

	return soap.ThumbprintSHA1(cert)
}
