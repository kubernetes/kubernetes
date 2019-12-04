// Copyright 2017 Google LLC
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

// Package option contains options for Google API clients.
package option

import (
	"net/http"

	"golang.org/x/oauth2"
	"google.golang.org/api/internal"
	"google.golang.org/grpc"
)

// A ClientOption is an option for a Google API client.
type ClientOption interface {
	Apply(*internal.DialSettings)
}

// WithTokenSource returns a ClientOption that specifies an OAuth2 token
// source to be used as the basis for authentication.
func WithTokenSource(s oauth2.TokenSource) ClientOption {
	return withTokenSource{s}
}

type withTokenSource struct{ ts oauth2.TokenSource }

func (w withTokenSource) Apply(o *internal.DialSettings) {
	o.TokenSource = w.ts
}

type withCredFile string

func (w withCredFile) Apply(o *internal.DialSettings) {
	o.CredentialsFile = string(w)
}

// WithCredentialsFile returns a ClientOption that authenticates
// API calls with the given service account or refresh token JSON
// credentials file.
func WithCredentialsFile(filename string) ClientOption {
	return withCredFile(filename)
}

// WithServiceAccountFile returns a ClientOption that uses a Google service
// account credentials file to authenticate.
//
// Deprecated: Use WithCredentialsFile instead.
func WithServiceAccountFile(filename string) ClientOption {
	return WithCredentialsFile(filename)
}

// WithCredentialsJSON returns a ClientOption that authenticates
// API calls with the given service account or refresh token JSON
// credentials.
func WithCredentialsJSON(p []byte) ClientOption {
	return withCredentialsJSON(p)
}

type withCredentialsJSON []byte

func (w withCredentialsJSON) Apply(o *internal.DialSettings) {
	o.CredentialsJSON = make([]byte, len(w))
	copy(o.CredentialsJSON, w)
}

// WithEndpoint returns a ClientOption that overrides the default endpoint
// to be used for a service.
func WithEndpoint(url string) ClientOption {
	return withEndpoint(url)
}

type withEndpoint string

func (w withEndpoint) Apply(o *internal.DialSettings) {
	o.Endpoint = string(w)
}

// WithScopes returns a ClientOption that overrides the default OAuth2 scopes
// to be used for a service.
func WithScopes(scope ...string) ClientOption {
	return withScopes(scope)
}

type withScopes []string

func (w withScopes) Apply(o *internal.DialSettings) {
	o.Scopes = make([]string, len(w))
	copy(o.Scopes, w)
}

// WithUserAgent returns a ClientOption that sets the User-Agent.
func WithUserAgent(ua string) ClientOption {
	return withUA(ua)
}

type withUA string

func (w withUA) Apply(o *internal.DialSettings) { o.UserAgent = string(w) }

// WithHTTPClient returns a ClientOption that specifies the HTTP client to use
// as the basis of communications. This option may only be used with services
// that support HTTP as their communication transport. When used, the
// WithHTTPClient option takes precedent over all other supplied options.
func WithHTTPClient(client *http.Client) ClientOption {
	return withHTTPClient{client}
}

type withHTTPClient struct{ client *http.Client }

func (w withHTTPClient) Apply(o *internal.DialSettings) {
	o.HTTPClient = w.client
}

// WithGRPCConn returns a ClientOption that specifies the gRPC client
// connection to use as the basis of communications. This option many only be
// used with services that support gRPC as their communication transport. When
// used, the WithGRPCConn option takes precedent over all other supplied
// options.
func WithGRPCConn(conn *grpc.ClientConn) ClientOption {
	return withGRPCConn{conn}
}

type withGRPCConn struct{ conn *grpc.ClientConn }

func (w withGRPCConn) Apply(o *internal.DialSettings) {
	o.GRPCConn = w.conn
}

// WithGRPCDialOption returns a ClientOption that appends a new grpc.DialOption
// to an underlying gRPC dial. It does not work with WithGRPCConn.
func WithGRPCDialOption(opt grpc.DialOption) ClientOption {
	return withGRPCDialOption{opt}
}

type withGRPCDialOption struct{ opt grpc.DialOption }

func (w withGRPCDialOption) Apply(o *internal.DialSettings) {
	o.GRPCDialOpts = append(o.GRPCDialOpts, w.opt)
}

// WithGRPCConnectionPool returns a ClientOption that creates a pool of gRPC
// connections that requests will be balanced between.
// This is an EXPERIMENTAL API and may be changed or removed in the future.
func WithGRPCConnectionPool(size int) ClientOption {
	return withGRPCConnectionPool(size)
}

type withGRPCConnectionPool int

func (w withGRPCConnectionPool) Apply(o *internal.DialSettings) {
	balancer := grpc.RoundRobin(internal.NewPoolResolver(int(w), o))
	o.GRPCDialOpts = append(o.GRPCDialOpts, grpc.WithBalancer(balancer))
}

// WithAPIKey returns a ClientOption that specifies an API key to be used
// as the basis for authentication.
//
// API Keys can only be used for JSON-over-HTTP APIs, including those under
// the import path google.golang.org/api/....
func WithAPIKey(apiKey string) ClientOption {
	return withAPIKey(apiKey)
}

type withAPIKey string

func (w withAPIKey) Apply(o *internal.DialSettings) { o.APIKey = string(w) }

// WithAudiences returns a ClientOption that specifies an audience to be used
// as the audience field ("aud") for the JWT token authentication.
func WithAudiences(audience ...string) ClientOption {
	return withAudiences(audience)
}

type withAudiences []string

func (w withAudiences) Apply(o *internal.DialSettings) {
	o.Audiences = make([]string, len(w))
	copy(o.Audiences, w)
}

// WithoutAuthentication returns a ClientOption that specifies that no
// authentication should be used. It is suitable only for testing and for
// accessing public resources, like public Google Cloud Storage buckets.
// It is an error to provide both WithoutAuthentication and any of WithAPIKey,
// WithTokenSource, WithCredentialsFile or WithServiceAccountFile.
func WithoutAuthentication() ClientOption {
	return withoutAuthentication{}
}

type withoutAuthentication struct{}

func (w withoutAuthentication) Apply(o *internal.DialSettings) { o.NoAuth = true }

// WithQuotaProject returns a ClientOption that specifies the project used
// for quota and billing purposes.
//
// For more information please read:
// https://cloud.google.com/apis/docs/system-parameters
func WithQuotaProject(quotaProject string) ClientOption {
	return withQuotaProject(quotaProject)
}

type withQuotaProject string

func (w withQuotaProject) Apply(o *internal.DialSettings) {
	o.QuotaProject = string(w)
}

// WithRequestReason returns a ClientOption that specifies a reason for
// making the request, which is intended to be recorded in audit logging.
// An example reason would be a support-case ticket number.
//
// For more information please read:
// https://cloud.google.com/apis/docs/system-parameters
func WithRequestReason(requestReason string) ClientOption {
	return withRequestReason(requestReason)
}

type withRequestReason string

func (w withRequestReason) Apply(o *internal.DialSettings) {
	o.RequestReason = string(w)
}
