// Copyright 2016 Google Inc. All Rights Reserved.
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

package cloud_test

import (
	"cloud.google.com/go/datastore"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
)

func Example_applicationDefaultCredentials() {
	ctx := context.Background()
	// Use Google Application Default Credentials to authorize and authenticate the client.
	// More information about Application Default Credentials and how to enable is at
	// https://developers.google.com/identity/protocols/application-default-credentials.
	//
	// This is the recommended way of authorizing and authenticating.
	//
	// Note: The example uses the datastore client, but the same steps apply to
	// the other client libraries underneath this package.
	client, err := datastore.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: handle error.
	}
	// Use the client.
	_ = client
}

func Example_serviceAccountFile() {
	// Warning: The better way to use service accounts is to set GOOGLE_APPLICATION_CREDENTIALS
	// and use the Application Default Credentials.
	ctx := context.Background()
	// Use a JSON key file associated with a Google service account to
	// authenticate and authorize.
	// Go to https://console.developers.google.com/permissions/serviceaccounts to create
	// and download a service account key for your project.
	//
	// Note: The example uses the datastore client, but the same steps apply to
	// the other client libraries underneath this package.
	client, err := datastore.NewClient(ctx,
		"project-id",
		option.WithServiceAccountFile("/path/to/service-account-key.json"))
	if err != nil {
		// TODO: handle error.
	}
	// Use the client.
	_ = client
}
