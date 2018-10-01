// Copyright 2018 Google Inc. All Rights Reserved.
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

// +build go1.9

package transport

import (
	"golang.org/x/net/context"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/internal"
	"google.golang.org/api/option"
)

// Creds constructs a google.Credentials from the information in the options,
// or obtains the default credentials in the same way as google.FindDefaultCredentials.
func Creds(ctx context.Context, opts ...option.ClientOption) (*google.Credentials, error) {
	var ds internal.DialSettings
	for _, opt := range opts {
		opt.Apply(&ds)
	}
	return internal.Creds(ctx, &ds)
}
