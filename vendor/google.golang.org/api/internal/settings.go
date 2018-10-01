// Copyright 2017 Google Inc. All Rights Reserved.
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

// Package internal supports the options and transport packages.
package internal

import (
	"errors"
	"net/http"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/grpc"
)

// DialSettings holds information needed to establish a connection with a
// Google API service.
type DialSettings struct {
	Endpoint        string
	Scopes          []string
	TokenSource     oauth2.TokenSource
	Credentials     *google.DefaultCredentials
	CredentialsFile string // if set, Token Source is ignored.
	UserAgent       string
	APIKey          string
	HTTPClient      *http.Client
	GRPCDialOpts    []grpc.DialOption
	GRPCConn        *grpc.ClientConn
	NoAuth          bool
}

// Validate reports an error if ds is invalid.
func (ds *DialSettings) Validate() error {
	hasCreds := ds.APIKey != "" || ds.TokenSource != nil || ds.CredentialsFile != "" || ds.Credentials != nil
	if ds.NoAuth && hasCreds {
		return errors.New("options.WithoutAuthentication is incompatible with any option that provides credentials")
	}
	// Credentials should not appear with other options.
	// We currently allow TokenSource and CredentialsFile to coexist.
	// TODO(jba): make TokenSource & CredentialsFile an error (breaking change).
	if ds.Credentials != nil && (ds.APIKey != "" || ds.TokenSource != nil || ds.CredentialsFile != "") {
		return errors.New("multiple credential options provided")
	}
	if ds.HTTPClient != nil && ds.GRPCConn != nil {
		return errors.New("WithHTTPClient is incompatible with WithGRPCConn")
	}
	if ds.HTTPClient != nil && ds.GRPCDialOpts != nil {
		return errors.New("WithHTTPClient is incompatible with gRPC dial options")
	}
	return nil
}
