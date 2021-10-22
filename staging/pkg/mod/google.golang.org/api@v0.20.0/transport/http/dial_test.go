// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"testing"

	"crypto/tls"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/api/internal"
)

func TestGetEndpoint(t *testing.T) {
	testCases := []struct {
		UserEndpoint    string
		DefaultEndpoint string
		Want            string
		WantErr         bool
	}{
		{
			DefaultEndpoint: "https://foo.googleapis.com/bar/baz",
			Want:            "https://foo.googleapis.com/bar/baz",
		},
		{
			UserEndpoint:    "myhost:3999",
			DefaultEndpoint: "https://foo.googleapis.com/bar/baz",
			Want:            "https://myhost:3999/bar/baz",
		},
		{
			UserEndpoint:    "https://host/path/to/bar",
			DefaultEndpoint: "https://foo.googleapis.com/bar/baz",
			Want:            "https://host/path/to/bar",
		},
		{
			UserEndpoint:    "host:port",
			DefaultEndpoint: "",
			WantErr:         true,
		},
	}

	for _, tc := range testCases {
		got, err := getEndpoint(&internal.DialSettings{
			Endpoint:        tc.UserEndpoint,
			DefaultEndpoint: tc.DefaultEndpoint,
		}, nil)
		if tc.WantErr && err == nil {
			t.Errorf("want err, got nil err")
			continue
		}
		if !tc.WantErr && err != nil {
			t.Errorf("want nil err, got %v", err)
			continue
		}
		if tc.Want != got {
			t.Errorf("getEndpoint(%q, %q): got %v; want %v", tc.UserEndpoint, tc.DefaultEndpoint, got, tc.Want)
		}
	}
}

func TestGetEndpointWithClientCertSource(t *testing.T) {
	dummyClientCertSource := func(info *tls.CertificateRequestInfo) (*tls.Certificate, error) { return nil, nil }
	testCases := []struct {
		UserEndpoint    string
		DefaultEndpoint string
		Want            string
		WantErr         bool
	}{
		{
			DefaultEndpoint: "https://foo.googleapis.com/bar/baz",
			Want:            "https://foo.mtls.googleapis.com/bar/baz",
		},
		{
			DefaultEndpoint: "https://staging-foo.sandbox.googleapis.com/bar/baz",
			Want:            "https://staging-foo.mtls.sandbox.googleapis.com/bar/baz",
		},
		{
			UserEndpoint:    "myhost:3999",
			DefaultEndpoint: "https://foo.googleapis.com/bar/baz",
			Want:            "https://myhost:3999/bar/baz",
		},
		{
			UserEndpoint:    "https://host/path/to/bar",
			DefaultEndpoint: "https://foo.googleapis.com/bar/baz",
			Want:            "https://host/path/to/bar",
		},
		{
			UserEndpoint:    "host:port",
			DefaultEndpoint: "",
			WantErr:         true,
		},
	}

	for _, tc := range testCases {
		got, err := getEndpoint(&internal.DialSettings{
			Endpoint:        tc.UserEndpoint,
			DefaultEndpoint: tc.DefaultEndpoint,
		}, dummyClientCertSource)
		if tc.WantErr && err == nil {
			t.Errorf("want err, got nil err")
			continue
		}
		if !tc.WantErr && err != nil {
			t.Errorf("want nil err, got %v", err)
			continue
		}
		if tc.Want != got {
			t.Errorf("getEndpoint(%q, %q): got %v; want %v", tc.UserEndpoint, tc.DefaultEndpoint, got, tc.Want)
		}
	}
}

func TestGenerateDefaultMtlsEndpoint(t *testing.T) {
	mtlsEndpoint := generateDefaultMtlsEndpoint("pubsub.googleapis.com")
	wantMtlsEndpoint := "pubsub.mtls.googleapis.com"
	if !cmp.Equal(mtlsEndpoint, wantMtlsEndpoint) {
		t.Error(cmp.Diff(wantMtlsEndpoint, wantMtlsEndpoint))
	}
}

func TestGenerateDefaultMtlsEndpointSandbox(t *testing.T) {
	mtlsEndpoint := generateDefaultMtlsEndpoint("staging-pubsub.sandbox.googleapis.com")
	wantMtlsEndpoint := "staging-pubsub.mtls.sandbox.googleapis.com"
	if !cmp.Equal(mtlsEndpoint, wantMtlsEndpoint) {
		t.Error(cmp.Diff(wantMtlsEndpoint, wantMtlsEndpoint))
	}
}

func TestGenerateDefaultMtlsEndpointUnsupported(t *testing.T) {
	mtlsEndpoint := generateDefaultMtlsEndpoint("unsupported.google.com")
	wantMtlsEndpoint := "unsupported.google.com"
	if !cmp.Equal(mtlsEndpoint, wantMtlsEndpoint) {
		t.Error(cmp.Diff(wantMtlsEndpoint, wantMtlsEndpoint))
	}
}
