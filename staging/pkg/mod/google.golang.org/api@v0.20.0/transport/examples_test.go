// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package transport_test

import (
	"context"
	"log"

	"google.golang.org/api/option"
	"google.golang.org/api/transport"
)

func Example_applicationDefaultCredentials() {
	ctx := context.Background()

	// Providing no auth option will cause NewClient to look for Application
	// Default Creds as specified at https://godoc.org/golang.org/x/oauth2/google#FindDefaultCredentials.
	//
	// Note: Given the same set of options, transport.NewHTTPClient and
	// transport.DialGRPC use the same credentials.
	c, _, err := transport.NewHTTPClient(ctx)
	if err != nil {
		log.Fatal(err)
	}
	_ = c // Use authenticated client.
}

func Example_withCredentialsFile() {
	ctx := context.Background()

	// Download service account creds per https://cloud.google.com/docs/authentication/production.
	//
	// Note: Given the same set of options, transport.NewHTTPClient and
	// transport.DialGRPC use the same credentials.
	c, _, err := transport.NewHTTPClient(ctx, option.WithCredentialsFile("/path/to/service-account-creds.json"))
	if err != nil {
		log.Fatal(err)
	}
	_ = c // Use authenticated client.
}
