// Copyright 2017 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internal supports the options and transport packages.
package internal

import (
	"crypto/tls"
	"errors"
	"net/http"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/internal/impersonate"
	"google.golang.org/grpc"
)

// DialSettings holds information needed to establish a connection with a
// Google API service.
type DialSettings struct {
	Endpoint                      string
	DefaultEndpoint               string
	DefaultMTLSEndpoint           string
	Scopes                        []string
	DefaultScopes                 []string
	EnableJwtWithScope            bool
	TokenSource                   oauth2.TokenSource
	Credentials                   *google.Credentials
	CredentialsFile               string // if set, Token Source is ignored.
	CredentialsJSON               []byte
	InternalCredentials           *google.Credentials
	UserAgent                     string
	APIKey                        string
	Audiences                     []string
	DefaultAudience               string
	HTTPClient                    *http.Client
	GRPCDialOpts                  []grpc.DialOption
	GRPCConn                      *grpc.ClientConn
	GRPCConnPool                  ConnPool
	GRPCConnPoolSize              int
	NoAuth                        bool
	TelemetryDisabled             bool
	ClientCertSource              func(*tls.CertificateRequestInfo) (*tls.Certificate, error)
	CustomClaims                  map[string]interface{}
	SkipValidation                bool
	ImpersonationConfig           *impersonate.Config
	EnableDirectPath              bool
	EnableDirectPathXds           bool
	AllowNonDefaultServiceAccount bool

	// Google API system parameters. For more information please read:
	// https://cloud.google.com/apis/docs/system-parameters
	QuotaProject  string
	RequestReason string
}

// GetScopes returns the user-provided scopes, if set, or else falls back to the
// default scopes.
func (ds *DialSettings) GetScopes() []string {
	if len(ds.Scopes) > 0 {
		return ds.Scopes
	}
	return ds.DefaultScopes
}

// GetAudience returns the user-provided audience, if set, or else falls back to the default audience.
func (ds *DialSettings) GetAudience() string {
	if ds.HasCustomAudience() {
		return ds.Audiences[0]
	}
	return ds.DefaultAudience
}

// HasCustomAudience returns true if a custom audience is provided by users.
func (ds *DialSettings) HasCustomAudience() bool {
	return len(ds.Audiences) > 0
}

// Validate reports an error if ds is invalid.
func (ds *DialSettings) Validate() error {
	if ds.SkipValidation {
		return nil
	}
	hasCreds := ds.APIKey != "" || ds.TokenSource != nil || ds.CredentialsFile != "" || ds.Credentials != nil
	if ds.NoAuth && hasCreds {
		return errors.New("options.WithoutAuthentication is incompatible with any option that provides credentials")
	}
	// Credentials should not appear with other options.
	// We currently allow TokenSource and CredentialsFile to coexist.
	// TODO(jba): make TokenSource & CredentialsFile an error (breaking change).
	nCreds := 0
	if ds.Credentials != nil {
		nCreds++
	}
	if ds.CredentialsJSON != nil {
		nCreds++
	}
	if ds.CredentialsFile != "" {
		nCreds++
	}
	if ds.APIKey != "" {
		nCreds++
	}
	if ds.TokenSource != nil {
		nCreds++
	}
	if len(ds.Scopes) > 0 && len(ds.Audiences) > 0 {
		return errors.New("WithScopes is incompatible with WithAudience")
	}
	// Accept only one form of credentials, except we allow TokenSource and CredentialsFile for backwards compatibility.
	if nCreds > 1 && !(nCreds == 2 && ds.TokenSource != nil && ds.CredentialsFile != "") {
		return errors.New("multiple credential options provided")
	}
	if ds.GRPCConn != nil && ds.GRPCConnPool != nil {
		return errors.New("WithGRPCConn is incompatible with WithConnPool")
	}
	if ds.HTTPClient != nil && ds.GRPCConnPool != nil {
		return errors.New("WithHTTPClient is incompatible with WithConnPool")
	}
	if ds.HTTPClient != nil && ds.GRPCConn != nil {
		return errors.New("WithHTTPClient is incompatible with WithGRPCConn")
	}
	if ds.HTTPClient != nil && ds.GRPCDialOpts != nil {
		return errors.New("WithHTTPClient is incompatible with gRPC dial options")
	}
	if ds.HTTPClient != nil && ds.QuotaProject != "" {
		return errors.New("WithHTTPClient is incompatible with QuotaProject")
	}
	if ds.HTTPClient != nil && ds.RequestReason != "" {
		return errors.New("WithHTTPClient is incompatible with RequestReason")
	}
	if ds.HTTPClient != nil && ds.ClientCertSource != nil {
		return errors.New("WithHTTPClient is incompatible with WithClientCertSource")
	}
	if ds.ClientCertSource != nil && (ds.GRPCConn != nil || ds.GRPCConnPool != nil || ds.GRPCConnPoolSize != 0 || ds.GRPCDialOpts != nil) {
		return errors.New("WithClientCertSource is currently only supported for HTTP. gRPC settings are incompatible")
	}
	if ds.ImpersonationConfig != nil && len(ds.ImpersonationConfig.Scopes) == 0 && len(ds.Scopes) == 0 {
		return errors.New("WithImpersonatedCredentials requires scopes being provided")
	}
	return nil
}
