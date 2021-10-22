// Copyright 2017 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package internal supports the options and transport packages.
package internal

import (
	"crypto/tls"
	"net/http"
	"testing"

	"google.golang.org/grpc"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

func TestSettingsValidate(t *testing.T) {
	dummyGetClientCertificate := func(info *tls.CertificateRequestInfo) (*tls.Certificate, error) { return nil, nil }

	// Valid.
	for _, ds := range []DialSettings{
		{},
		{APIKey: "x"},
		{Scopes: []string{"s"}},
		{CredentialsFile: "f"},
		{TokenSource: dummyTS{}},
		{CredentialsFile: "f", TokenSource: dummyTS{}}, // keep for backwards compatibility
		{CredentialsJSON: []byte("json")},
		{HTTPClient: &http.Client{}},
		{GRPCConn: &grpc.ClientConn{}},
		// Although NoAuth and Scopes are technically incompatible, too many
		// cloud clients add WithScopes to user-provided options to make
		// the check feasible.
		{NoAuth: true, Scopes: []string{"s"}},
		{ClientCertSource: dummyGetClientCertificate},
	} {
		err := ds.Validate()
		if err != nil {
			t.Errorf("%+v: got %v, want nil", ds, err)
		}
	}

	// Invalid.
	for _, ds := range []DialSettings{
		{NoAuth: true, APIKey: "x"},
		{NoAuth: true, CredentialsFile: "f"},
		{NoAuth: true, TokenSource: dummyTS{}},
		{NoAuth: true, Credentials: &google.DefaultCredentials{}},
		{Credentials: &google.DefaultCredentials{}, CredentialsFile: "f"},
		{Credentials: &google.DefaultCredentials{}, TokenSource: dummyTS{}},
		{Credentials: &google.DefaultCredentials{}, CredentialsJSON: []byte("json")},
		{CredentialsFile: "f", CredentialsJSON: []byte("json")},
		{CredentialsJSON: []byte("json"), TokenSource: dummyTS{}},
		{HTTPClient: &http.Client{}, GRPCConn: &grpc.ClientConn{}},
		{HTTPClient: &http.Client{}, GRPCDialOpts: []grpc.DialOption{grpc.WithInsecure()}},
		{Audiences: []string{"foo"}, Scopes: []string{"foo"}},
		{HTTPClient: &http.Client{}, QuotaProject: "foo"},
		{HTTPClient: &http.Client{}, RequestReason: "foo"},
		{HTTPClient: &http.Client{}, ClientCertSource: dummyGetClientCertificate},
		{ClientCertSource: dummyGetClientCertificate, GRPCConn: &grpc.ClientConn{}},
		{ClientCertSource: dummyGetClientCertificate, GRPCConnPool: struct{ ConnPool }{}},
		{ClientCertSource: dummyGetClientCertificate, GRPCDialOpts: []grpc.DialOption{grpc.WithInsecure()}},
		{ClientCertSource: dummyGetClientCertificate, GRPCConnPoolSize: 1},
	} {
		err := ds.Validate()
		if err == nil {
			t.Errorf("%+v: got nil, want error", ds)
		}
	}

}

type dummyTS struct{}

func (dummyTS) Token() (*oauth2.Token, error) { return nil, nil }
