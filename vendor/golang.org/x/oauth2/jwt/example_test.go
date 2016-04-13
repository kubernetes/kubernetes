// Copyright 2014 The oauth2 Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package jwt_test

import (
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/jwt"
)

func ExampleJWTConfig() {
	conf := &jwt.Config{
		Email: "xxx@developer.com",
		// The contents of your RSA private key or your PEM file
		// that contains a private key.
		// If you have a p12 file instead, you
		// can use `openssl` to export the private key into a pem file.
		//
		//    $ openssl pkcs12 -in key.p12 -out key.pem -nodes
		//
		// It only supports PEM containers with no passphrase.
		PrivateKey: []byte("-----BEGIN RSA PRIVATE KEY-----..."),
		Subject:    "user@example.com",
		TokenURL:   "https://provider.com/o/oauth2/token",
	}
	// Initiate an http.Client, the following GET request will be
	// authorized and authenticated on the behalf of user@example.com.
	client := conf.Client(oauth2.NoContext)
	client.Get("...")
}
