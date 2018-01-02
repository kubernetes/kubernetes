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
	"net/http"

	"golang.org/x/oauth2"
	"google.golang.org/grpc"
)

// DialSettings holds information needed to establish a connection with a
// Google API service.
type DialSettings struct {
	Endpoint        string
	Scopes          []string
	TokenSource     oauth2.TokenSource
	CredentialsFile string // if set, Token Source is ignored.
	UserAgent       string
	APIKey          string
	HTTPClient      *http.Client
	GRPCDialOpts    []grpc.DialOption
	GRPCConn        *grpc.ClientConn
}
