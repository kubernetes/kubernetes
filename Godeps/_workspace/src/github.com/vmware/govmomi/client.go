/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

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

/*
This package is the root package of the govmomi library.

The library is structured as follows:

Package vim25

The minimal usable functionality is available through the vim25 package.
It contains subpackages that contain generated types, managed objects, and all
available methods. The vim25 package is entirely independent of the other
packages in the govmomi tree -- it has no dependencies on its peers.

The vim25 package itself contains a client structure that is
passed around throughout the entire library. It abstracts a session and its
immutable state. See the vim25 package for more information.

Package session

The session package contains an abstraction for the session manager that allows
a user to login and logout. It also provides access to the current session
(i.e. to determine if the user is in fact logged in)

Package object

The object package contains wrappers for a selection of managed objects. The
constructors of these objects all take a *vim25.Client, which they pass along
to derived objects, if applicable.

Package govc

The govc package contains the govc CLI. The code in this tree is not intended
to be used as a library. Any functionality that govc contains that _could_ be
used as a library function but isn't, _should_ live in a root level package.

Other packages

Other packages, such as "event", "guest", or "license", provide wrappers for
the respective subsystems. They are typically not needed in normal workflows so
are kept outside the object package.
*/
package govmomi

import (
	"crypto/tls"
	"net/url"

	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/session"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
	"golang.org/x/net/context"
)

type Client struct {
	*vim25.Client

	SessionManager *session.Manager
}

// NewClient creates a new client from a URL. The client authenticates with the
// server with username/password before returning if the URL contains user information.
func NewClient(ctx context.Context, u *url.URL, insecure bool) (*Client, error) {
	soapClient := soap.NewClient(u, insecure)
	vimClient, err := vim25.NewClient(ctx, soapClient)
	if err != nil {
		return nil, err
	}

	c := &Client{
		Client:         vimClient,
		SessionManager: session.NewManager(vimClient),
	}

	// Only login if the URL contains user information.
	if u.User != nil {
		err = c.Login(ctx, u.User)
		if err != nil {
			return nil, err
		}
	}

	return c, nil
}

// NewClientWithCertificate creates a new client from a URL. The client authenticates with the
// server with the certificate before returning if the URL contains user information.
func NewClientWithCertificate(ctx context.Context, u *url.URL, insecure bool, cert tls.Certificate) (*Client, error) {
	soapClient := soap.NewClient(u, insecure)
	soapClient.SetCertificate(cert)
	vimClient, err := vim25.NewClient(ctx, soapClient)
	if err != nil {
		return nil, err
	}

	c := &Client{
		Client:         vimClient,
		SessionManager: session.NewManager(vimClient),
	}

	if u.User != nil {
		err = c.LoginExtensionByCertificate(ctx, u.User.Username(), "")
		if err != nil {
			return nil, err
		}
	}

	return c, nil
}

// Login dispatches to the SessionManager.
func (c *Client) Login(ctx context.Context, u *url.Userinfo) error {
	return c.SessionManager.Login(ctx, u)
}

// Login dispatches to the SessionManager.
func (c *Client) LoginExtensionByCertificate(ctx context.Context, key string, locale string) error {
	return c.SessionManager.LoginExtensionByCertificate(ctx, key, locale)
}

// Logout dispatches to the SessionManager.
func (c *Client) Logout(ctx context.Context) error {
	// Close any idle connections after logging out.
	defer c.Client.CloseIdleConnections()
	return c.SessionManager.Logout(ctx)
}

// PropertyCollector returns the session's default property collector.
func (c *Client) PropertyCollector() *property.Collector {
	return property.DefaultCollector(c.Client)
}

// RetrieveOne dispatches to the Retrieve function on the default property collector.
func (c *Client) RetrieveOne(ctx context.Context, obj types.ManagedObjectReference, p []string, dst interface{}) error {
	return c.PropertyCollector().RetrieveOne(ctx, obj, p, dst)
}

// Retrieve dispatches to the Retrieve function on the default property collector.
func (c *Client) Retrieve(ctx context.Context, objs []types.ManagedObjectReference, p []string, dst interface{}) error {
	return c.PropertyCollector().Retrieve(ctx, objs, p, dst)
}

// Wait dispatches to property.Wait.
func (c *Client) Wait(ctx context.Context, obj types.ManagedObjectReference, ps []string, f func([]types.PropertyChange) bool) error {
	return property.Wait(ctx, c.PropertyCollector(), obj, ps, f)
}

// IsVC returns true if we are connected to a vCenter
func (c *Client) IsVC() bool {
	return c.Client.IsVC()
}
