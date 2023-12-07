// Copyright 2017 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package option contains options for Google API clients.
package option

import (
	"crypto/tls"
	"net/http"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/internal"
	"google.golang.org/api/internal/impersonate"
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
//
// If both WithScopes and WithTokenSource are used, scope settings from the
// token source will be used instead.
func WithScopes(scope ...string) ClientOption {
	return withScopes(scope)
}

type withScopes []string

func (w withScopes) Apply(o *internal.DialSettings) {
	o.Scopes = make([]string, len(w))
	copy(o.Scopes, w)
}

// WithUserAgent returns a ClientOption that sets the User-Agent. This option
// is incompatible with the [WithHTTPClient] option. If you wish to provide a
// custom client you will need to add this header via RoundTripper middleware.
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
// connection to use as the basis of communications. This option may only be
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
func WithGRPCConnectionPool(size int) ClientOption {
	return withGRPCConnectionPool(size)
}

type withGRPCConnectionPool int

func (w withGRPCConnectionPool) Apply(o *internal.DialSettings) {
	o.GRPCConnPoolSize = int(w)
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

// WithTelemetryDisabled returns a ClientOption that disables default telemetry (OpenCensus)
// settings on gRPC and HTTP clients.
// An example reason would be to bind custom telemetry that overrides the defaults.
func WithTelemetryDisabled() ClientOption {
	return withTelemetryDisabled{}
}

type withTelemetryDisabled struct{}

func (w withTelemetryDisabled) Apply(o *internal.DialSettings) {
	o.TelemetryDisabled = true
}

// ClientCertSource is a function that returns a TLS client certificate to be used
// when opening TLS connections.
//
// It follows the same semantics as crypto/tls.Config.GetClientCertificate.
//
// This is an EXPERIMENTAL API and may be changed or removed in the future.
type ClientCertSource = func(*tls.CertificateRequestInfo) (*tls.Certificate, error)

// WithClientCertSource returns a ClientOption that specifies a
// callback function for obtaining a TLS client certificate.
//
// This option is used for supporting mTLS authentication, where the
// server validates the client certifcate when establishing a connection.
//
// The callback function will be invoked whenever the server requests a
// certificate from the client. Implementations of the callback function
// should try to ensure that a valid certificate can be repeatedly returned
// on demand for the entire life cycle of the transport client. If a nil
// Certificate is returned (i.e. no Certificate can be obtained), an error
// should be returned.
//
// This is an EXPERIMENTAL API and may be changed or removed in the future.
func WithClientCertSource(s ClientCertSource) ClientOption {
	return withClientCertSource{s}
}

type withClientCertSource struct{ s ClientCertSource }

func (w withClientCertSource) Apply(o *internal.DialSettings) {
	o.ClientCertSource = w.s
}

// ImpersonateCredentials returns a ClientOption that will impersonate the
// target service account.
//
// In order to impersonate the target service account
// the base service account must have the Service Account Token Creator role,
// roles/iam.serviceAccountTokenCreator, on the target service account.
// See https://cloud.google.com/iam/docs/understanding-service-accounts.
//
// Optionally, delegates can be used during impersonation if the base service
// account lacks the token creator role on the target. When using delegates,
// each service account must be granted roles/iam.serviceAccountTokenCreator
// on the next service account in the chain.
//
// For example, if a base service account of SA1 is trying to impersonate target
// service account SA2 while using delegate service accounts DSA1 and DSA2,
// the following must be true:
//
//  1. Base service account SA1 has roles/iam.serviceAccountTokenCreator on
//     DSA1.
//  2. DSA1 has roles/iam.serviceAccountTokenCreator on DSA2.
//  3. DSA2 has roles/iam.serviceAccountTokenCreator on target SA2.
//
// The resulting impersonated credential will either have the default scopes of
// the client being instantiating or the scopes from WithScopes if provided.
// Scopes are required for creating impersonated credentials, so if this option
// is used while not using a NewClient/NewService function, WithScopes must also
// be explicitly passed in as well.
//
// If the base credential is an authorized user and not a service account, or if
// the option WithQuotaProject is set, the target service account must have a
// role that grants the serviceusage.services.use permission such as
// roles/serviceusage.serviceUsageConsumer.
//
// This is an EXPERIMENTAL API and may be changed or removed in the future.
//
// Deprecated: This option has been replaced by `impersonate` package:
// `google.golang.org/api/impersonate`. Please use the `impersonate` package
// instead with the WithTokenSource option.
func ImpersonateCredentials(target string, delegates ...string) ClientOption {
	return impersonateServiceAccount{
		target:    target,
		delegates: delegates,
	}
}

type impersonateServiceAccount struct {
	target    string
	delegates []string
}

func (i impersonateServiceAccount) Apply(o *internal.DialSettings) {
	o.ImpersonationConfig = &impersonate.Config{
		Target: i.target,
	}
	o.ImpersonationConfig.Delegates = make([]string, len(i.delegates))
	copy(o.ImpersonationConfig.Delegates, i.delegates)
}

type withCreds google.Credentials

func (w *withCreds) Apply(o *internal.DialSettings) {
	o.Credentials = (*google.Credentials)(w)
}

// WithCredentials returns a ClientOption that authenticates API calls.
func WithCredentials(creds *google.Credentials) ClientOption {
	return (*withCreds)(creds)
}
