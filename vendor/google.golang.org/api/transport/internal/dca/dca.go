// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dca contains utils for implementing Device Certificate
// Authentication according to https://google.aip.dev/auth/4114
//
// The overall logic for DCA is as follows:
// 1. If both endpoint override and client certificate are specified, use them as is.
// 2. If user does not specify client certificate, we will attempt to use default
//    client certificate.
// 3. If user does not specify endpoint override, we will use defaultMtlsEndpoint if
//    client certificate is available and defaultEndpoint otherwise.
//
// Implications of the above logic:
// 1. If the user specifies a non-mTLS endpoint override but client certificate is
//    available, we will pass along the cert anyway and let the server decide what to do.
// 2. If the user specifies an mTLS endpoint override but client certificate is not
//    available, we will not fail-fast, but let backend throw error when connecting.
//
// We would like to avoid introducing client-side logic that parses whether the
// endpoint override is an mTLS url, since the url pattern may change at anytime.
//
// This package is not intended for use by end developers. Use the
// google.golang.org/api/option package to configure API clients.
package dca

import (
	"net/url"
	"os"
	"strings"

	"google.golang.org/api/internal"
	"google.golang.org/api/transport/cert"
)

const (
	mTLSModeAlways = "always"
	mTLSModeNever  = "never"
	mTLSModeAuto   = "auto"
)

// GetClientCertificateSourceAndEndpoint is a convenience function that invokes
// getClientCertificateSource and getEndpoint sequentially and returns the client
// cert source and endpoint as a tuple.
func GetClientCertificateSourceAndEndpoint(settings *internal.DialSettings) (cert.Source, string, error) {
	clientCertSource, err := getClientCertificateSource(settings)
	if err != nil {
		return nil, "", err
	}
	endpoint, err := getEndpoint(settings, clientCertSource)
	if err != nil {
		return nil, "", err
	}
	return clientCertSource, endpoint, nil
}

// getClientCertificateSource returns a default client certificate source, if
// not provided by the user.
//
// A nil default source can be returned if the source does not exist. Any exceptions
// encountered while initializing the default source will be reported as client
// error (ex. corrupt metadata file).
//
// Important Note: For now, the environment variable GOOGLE_API_USE_CLIENT_CERTIFICATE
// must be set to "true" to allow certificate to be used (including user provided
// certificates). For details, see AIP-4114.
func getClientCertificateSource(settings *internal.DialSettings) (cert.Source, error) {
	if !isClientCertificateEnabled() {
		return nil, nil
	} else if settings.ClientCertSource != nil {
		return settings.ClientCertSource, nil
	} else {
		return cert.DefaultSource()
	}
}

func isClientCertificateEnabled() bool {
	useClientCert := os.Getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE")
	// TODO(andyrzhao): Update default to return "true" after DCA feature is fully released.
	return strings.ToLower(useClientCert) == "true"
}

// getEndpoint returns the endpoint for the service, taking into account the
// user-provided endpoint override "settings.Endpoint".
//
// If no endpoint override is specified, we will either return the default endpoint or
// the default mTLS endpoint if a client certificate is available.
//
// You can override the default endpoint choice (mtls vs. regular) by setting the
// GOOGLE_API_USE_MTLS_ENDPOINT environment variable.
//
// If the endpoint override is an address (host:port) rather than full base
// URL (ex. https://...), then the user-provided address will be merged into
// the default endpoint. For example, WithEndpoint("myhost:8000") and
// WithDefaultEndpoint("https://foo.com/bar/baz") will return "https://myhost:8080/bar/baz"
func getEndpoint(settings *internal.DialSettings, clientCertSource cert.Source) (string, error) {
	if settings.Endpoint == "" {
		mtlsMode := getMTLSMode()
		if mtlsMode == mTLSModeAlways || (clientCertSource != nil && mtlsMode == mTLSModeAuto) {
			return settings.DefaultMTLSEndpoint, nil
		}
		return settings.DefaultEndpoint, nil
	}
	if strings.Contains(settings.Endpoint, "://") {
		// User passed in a full URL path, use it verbatim.
		return settings.Endpoint, nil
	}
	if settings.DefaultEndpoint == "" {
		// If DefaultEndpoint is not configured, use the user provided endpoint verbatim.
		// This allows a naked "host[:port]" URL to be used with GRPC Direct Path.
		return settings.Endpoint, nil
	}

	// Assume user-provided endpoint is host[:port], merge it with the default endpoint.
	return mergeEndpoints(settings.DefaultEndpoint, settings.Endpoint)
}

func getMTLSMode() string {
	mode := os.Getenv("GOOGLE_API_USE_MTLS_ENDPOINT")
	if mode == "" {
		mode = os.Getenv("GOOGLE_API_USE_MTLS") // Deprecated.
	}
	if mode == "" {
		return mTLSModeAuto
	}
	return strings.ToLower(mode)
}

func mergeEndpoints(baseURL, newHost string) (string, error) {
	u, err := url.Parse(fixScheme(baseURL))
	if err != nil {
		return "", err
	}
	return strings.Replace(baseURL, u.Host, newHost, 1), nil
}

func fixScheme(baseURL string) string {
	if !strings.Contains(baseURL, "://") {
		return "https://" + baseURL
	}
	return baseURL
}
