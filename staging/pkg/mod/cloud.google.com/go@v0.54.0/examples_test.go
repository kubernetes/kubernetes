// Copyright 2018 Google LLC
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
	"context"
	"time"

	"cloud.google.com/go/bigquery"
	"cloud.google.com/go/datastore"
	"cloud.google.com/go/pubsub"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/option"
)

// To set a timeout for an RPC, use context.WithTimeout.
func Example_timeout() {
	ctx := context.Background()
	// Do not set a timeout on the context passed to NewClient: dialing happens
	// asynchronously, and the context is used to refresh credentials in the
	// background.
	client, err := bigquery.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: handle error.
	}
	// Time out if it takes more than 10 seconds to create a dataset.
	tctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel() // Always call cancel.

	if err := client.Dataset("new-dataset").Create(tctx, nil); err != nil {
		// TODO: handle error.
	}
}

// To arrange for an RPC to be canceled, use context.WithCancel.
func Example_cancellation() {
	ctx := context.Background()
	// Do not cancel the context passed to NewClient: dialing happens asynchronously,
	// and the context is used to refresh credentials in the background.
	client, err := bigquery.NewClient(ctx, "project-id")
	if err != nil {
		// TODO: handle error.
	}
	cctx, cancel := context.WithCancel(ctx)
	defer cancel() // Always call cancel.

	// TODO: Make the cancel function available to whatever might want to cancel the
	// call--perhaps a GUI button.
	if err := client.Dataset("new-dataset").Create(cctx, nil); err != nil {
		// TODO: handle error.
	}
}

// Google Application Default Credentials is the recommended way to authorize
// and authenticate clients.
//
// For information on how to create and obtain Application Default Credentials, see
// https://developers.google.com/identity/protocols/application-default-credentials.
func Example_applicationDefaultCredentials() {
	client, err := datastore.NewClient(context.Background(), "project-id")
	if err != nil {
		// TODO: handle error.
	}
	_ = client // Use the client.
}

// You can use a file with credentials to authenticate and authorize, such as a JSON
// key file associated with a Google service account. Service Account keys can be
// created and downloaded from
// https://console.developers.google.com/permissions/serviceaccounts.
//
// This example uses the Datastore client, but the same steps apply to
// the other client libraries underneath this package.
func Example_credentialsFile() {
	client, err := datastore.NewClient(context.Background(),
		"project-id", option.WithCredentialsFile("/path/to/service-account-key.json"))
	if err != nil {
		// TODO: handle error.
	}
	_ = client // Use the client.
}

// In some cases (for instance, you don't want to store secrets on disk), you can
// create credentials from in-memory JSON and use the WithCredentials option.
//
// The google package in this example is at golang.org/x/oauth2/google.
//
// This example uses the PubSub client, but the same steps apply to
// the other client libraries underneath this package. Note that scopes can be
// found at https://developers.google.com/identity/protocols/googlescopes, and
// are also provided in all auto-generated libraries: for example,
// cloud.google.com/go/pubsub/apiv1 provides DefaultAuthScopes.
func Example_credentialsFromJSON() {
	ctx := context.Background()
	creds, err := google.CredentialsFromJSON(ctx, []byte("JSON creds"), pubsub.ScopePubSub)
	if err != nil {
		// TODO: handle error.
	}
	client, err := pubsub.NewClient(ctx, "project-id", option.WithCredentials(creds))
	if err != nil {
		// TODO: handle error.
	}
	_ = client // Use the client.
}
