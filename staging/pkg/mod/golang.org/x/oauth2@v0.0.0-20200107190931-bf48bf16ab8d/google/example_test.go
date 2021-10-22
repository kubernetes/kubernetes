// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google_test

import (
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"golang.org/x/oauth2/jwt"
)

func ExampleDefaultClient() {
	client, err := google.DefaultClient(oauth2.NoContext,
		"https://www.googleapis.com/auth/devstorage.full_control")
	if err != nil {
		log.Fatal(err)
	}
	client.Get("...")
}

func Example_webServer() {
	// Your credentials should be obtained from the Google
	// Developer Console (https://console.developers.google.com).
	conf := &oauth2.Config{
		ClientID:     "YOUR_CLIENT_ID",
		ClientSecret: "YOUR_CLIENT_SECRET",
		RedirectURL:  "YOUR_REDIRECT_URL",
		Scopes: []string{
			"https://www.googleapis.com/auth/bigquery",
			"https://www.googleapis.com/auth/blogger",
		},
		Endpoint: google.Endpoint,
	}
	// Redirect user to Google's consent page to ask for permission
	// for the scopes specified above.
	url := conf.AuthCodeURL("state")
	fmt.Printf("Visit the URL for the auth dialog: %v", url)

	// Handle the exchange code to initiate a transport.
	tok, err := conf.Exchange(oauth2.NoContext, "authorization-code")
	if err != nil {
		log.Fatal(err)
	}
	client := conf.Client(oauth2.NoContext, tok)
	client.Get("...")
}

func ExampleJWTConfigFromJSON() {
	// Your credentials should be obtained from the Google
	// Developer Console (https://console.developers.google.com).
	// Navigate to your project, then see the "Credentials" page
	// under "APIs & Auth".
	// To create a service account client, click "Create new Client ID",
	// select "Service Account", and click "Create Client ID". A JSON
	// key file will then be downloaded to your computer.
	data, err := ioutil.ReadFile("/path/to/your-project-key.json")
	if err != nil {
		log.Fatal(err)
	}
	conf, err := google.JWTConfigFromJSON(data, "https://www.googleapis.com/auth/bigquery")
	if err != nil {
		log.Fatal(err)
	}
	// Initiate an http.Client. The following GET request will be
	// authorized and authenticated on the behalf of
	// your service account.
	client := conf.Client(oauth2.NoContext)
	client.Get("...")
}

func ExampleSDKConfig() {
	// The credentials will be obtained from the first account that
	// has been authorized with `gcloud auth login`.
	conf, err := google.NewSDKConfig("")
	if err != nil {
		log.Fatal(err)
	}
	// Initiate an http.Client. The following GET request will be
	// authorized and authenticated on the behalf of the SDK user.
	client := conf.Client(oauth2.NoContext)
	client.Get("...")
}

func Example_serviceAccount() {
	// Your credentials should be obtained from the Google
	// Developer Console (https://console.developers.google.com).
	conf := &jwt.Config{
		Email: "xxx@developer.gserviceaccount.com",
		// The contents of your RSA private key or your PEM file
		// that contains a private key.
		// If you have a p12 file instead, you
		// can use `openssl` to export the private key into a pem file.
		//
		//    $ openssl pkcs12 -in key.p12 -passin pass:notasecret -out key.pem -nodes
		//
		// The field only supports PEM containers with no passphrase.
		// The openssl command will convert p12 keys to passphrase-less PEM containers.
		PrivateKey: []byte("-----BEGIN RSA PRIVATE KEY-----..."),
		Scopes: []string{
			"https://www.googleapis.com/auth/bigquery",
			"https://www.googleapis.com/auth/blogger",
		},
		TokenURL: google.JWTTokenURL,
		// If you would like to impersonate a user, you can
		// create a transport with a subject. The following GET
		// request will be made on the behalf of user@example.com.
		// Optional.
		Subject: "user@example.com",
	}
	// Initiate an http.Client, the following GET request will be
	// authorized and authenticated on the behalf of user@example.com.
	client := conf.Client(oauth2.NoContext)
	client.Get("...")
}

func ExampleComputeTokenSource() {
	client := &http.Client{
		Transport: &oauth2.Transport{
			// Fetch from Google Compute Engine's metadata server to retrieve
			// an access token for the provided account.
			// If no account is specified, "default" is used.
			// If no scopes are specified, a set of default scopes
			// are automatically granted.
			Source: google.ComputeTokenSource("", "https://www.googleapis.com/auth/bigquery"),
		},
	}
	client.Get("...")
}

func ExampleCredentialsFromJSON() {
	ctx := context.Background()
	data, err := ioutil.ReadFile("/path/to/key-file.json")
	if err != nil {
		log.Fatal(err)
	}
	creds, err := google.CredentialsFromJSON(ctx, data, "https://www.googleapis.com/auth/bigquery")
	if err != nil {
		log.Fatal(err)
	}
	_ = creds // TODO: Use creds.
}
