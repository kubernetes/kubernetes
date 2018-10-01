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

package internal

import (
	"fmt"
	"io/ioutil"

	"golang.org/x/net/context"
	"golang.org/x/oauth2/google"
)

// Creds returns credential information obtained from DialSettings, or if none, then
// it returns default credential information.
func Creds(ctx context.Context, ds *DialSettings) (*google.DefaultCredentials, error) {
	if ds.Credentials != nil {
		return ds.Credentials, nil
	}
	if ds.CredentialsFile != "" {
		data, err := ioutil.ReadFile(ds.CredentialsFile)
		if err != nil {
			return nil, fmt.Errorf("cannot read credentials file: %v", err)
		}
		return google.CredentialsFromJSON(ctx, data, ds.Scopes...)
	}
	if ds.TokenSource != nil {
		return &google.DefaultCredentials{TokenSource: ds.TokenSource}, nil
	}
	return google.FindDefaultCredentials(ctx, ds.Scopes...)
}
