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

package sts

import (
	"context"
	"crypto/tls"
	"errors"
	"net/url"
	"time"

	"github.com/vmware/govmomi/lookup"
	"github.com/vmware/govmomi/lookup/types"
	"github.com/vmware/govmomi/sts/internal"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
)

const (
	Namespace = "oasis:names:tc:SAML:2.0:assertion"
	Path      = "/sts/STSService"
)

// Client is a soap.Client targeting the STS (Secure Token Service) API endpoint.
type Client struct {
	*soap.Client
}

// NewClient returns a client targeting the STS API endpoint.
// The Client.URL will be set to that of the Lookup Service's endpoint registration,
// as the SSO endpoint can be external to vCenter.  If the Lookup Service is not available,
// URL defaults to Path on the vim25.Client.URL.Host.
func NewClient(ctx context.Context, c *vim25.Client) (*Client, error) {
	filter := &types.LookupServiceRegistrationFilter{
		ServiceType: &types.LookupServiceRegistrationServiceType{
			Product: "com.vmware.cis",
			Type:    "sso:sts",
		},
		EndpointType: &types.LookupServiceRegistrationEndpointType{
			Protocol: "wsTrust",
			Type:     "com.vmware.cis.cs.identity.sso",
		},
	}

	url := lookup.EndpointURL(ctx, c, Path, filter)
	sc := c.Client.NewServiceClient(url, Namespace)

	return &Client{sc}, nil
}

// TokenRequest parameters for issuing a SAML token.
// At least one of Userinfo or Certificate must be specified.
type TokenRequest struct {
	Userinfo    *url.Userinfo    // Userinfo when set issues a Bearer token
	Certificate *tls.Certificate // Certificate when set issues a HoK token
	Lifetime    time.Duration    // Lifetime is the token's lifetime, defaults to 10m
	Renewable   bool             // Renewable allows the issued token to be renewed
	Delegatable bool             // Delegatable allows the issued token to be delegated (e.g. for use with ActAs)
	Token       string           // Token for Renew request or Issue request ActAs identity
}

func (c *Client) newRequest(req TokenRequest, kind string, s *Signer) (internal.RequestSecurityToken, error) {
	if req.Lifetime == 0 {
		req.Lifetime = 5 * time.Minute
	}

	created := time.Now().UTC()
	rst := internal.RequestSecurityToken{
		TokenType:          c.Namespace,
		RequestType:        "http://docs.oasis-open.org/ws-sx/ws-trust/200512/" + kind,
		SignatureAlgorithm: internal.SHA256,
		Lifetime: &internal.Lifetime{
			Created: created.Format(internal.Time),
			Expires: created.Add(req.Lifetime).Format(internal.Time),
		},
		Renewing: &internal.Renewing{
			Allow: req.Renewable,
			// /wst:RequestSecurityToken/wst:Renewing/@OK
			// "It NOT RECOMMENDED to use this as it can leave you open to certain types of security attacks.
			// Issuers MAY restrict the period after expiration during which time the token can be renewed.
			// This window is governed by the issuer's policy."
			OK: false,
		},
		Delegatable: req.Delegatable,
	}

	if req.Certificate == nil {
		if req.Userinfo == nil {
			return rst, errors.New("one of TokenRequest Certificate or Userinfo is required")
		}
		rst.KeyType = "http://docs.oasis-open.org/ws-sx/ws-trust/200512/Bearer"
	} else {
		rst.KeyType = "http://docs.oasis-open.org/ws-sx/ws-trust/200512/PublicKey"
		rst.UseKey = &internal.UseKey{Sig: newID()}
		s.keyID = rst.UseKey.Sig
	}

	return rst, nil
}

func (s *Signer) setLifetime(lifetime *internal.Lifetime) error {
	var err error
	if lifetime != nil {
		s.Lifetime.Created, err = time.Parse(internal.Time, lifetime.Created)
		if err == nil {
			s.Lifetime.Expires, err = time.Parse(internal.Time, lifetime.Expires)
		}
	}
	return err
}

// Issue is used to request a security token.
// The returned Signer can be used to sign SOAP requests, such as the SessionManager LoginByToken method and the RequestSecurityToken method itself.
// One of TokenRequest Certificate or Userinfo is required, with Certificate taking precedence.
// When Certificate is set, a Holder-of-Key token will be requested.  Otherwise, a Bearer token is requested with the Userinfo credentials.
// See: http://docs.oasis-open.org/ws-sx/ws-trust/v1.4/errata01/os/ws-trust-1.4-errata01-os-complete.html#_Toc325658937
func (c *Client) Issue(ctx context.Context, req TokenRequest) (*Signer, error) {
	s := &Signer{
		Certificate: req.Certificate,
		user:        req.Userinfo,
	}

	rst, err := c.newRequest(req, "Issue", s)
	if err != nil {
		return nil, err
	}

	if req.Token != "" {
		rst.ActAs = &internal.Target{
			Token: req.Token,
		}
	}

	header := soap.Header{
		Security: s,
		Action:   rst.Action(),
	}

	res, err := internal.Issue(c.WithHeader(ctx, header), c, &rst)
	if err != nil {
		return nil, err
	}

	s.Token = res.RequestSecurityTokenResponse.RequestedSecurityToken.Assertion

	return s, s.setLifetime(res.RequestSecurityTokenResponse.Lifetime)
}

// Renew is used to request a security token renewal.
func (c *Client) Renew(ctx context.Context, req TokenRequest) (*Signer, error) {
	s := &Signer{
		Certificate: req.Certificate,
	}

	rst, err := c.newRequest(req, "Renew", s)
	if err != nil {
		return nil, err
	}

	if req.Token == "" {
		return nil, errors.New("TokenRequest Token is required")
	}

	rst.RenewTarget = &internal.Target{Token: req.Token}

	header := soap.Header{
		Security: s,
		Action:   rst.Action(),
	}

	res, err := internal.Renew(c.WithHeader(ctx, header), c, &rst)
	if err != nil {
		return nil, err
	}

	s.Token = res.RequestedSecurityToken.Assertion

	return s, s.setLifetime(res.Lifetime)
}
