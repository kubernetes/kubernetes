// Copyright 2020 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cba.go (certificate-based access) contains utils for implementing Device Certificate
// Authentication according to https://google.aip.dev/auth/4114 and Default Credentials
// for Google Cloud Virtual Environments according to https://google.aip.dev/auth/4115.
//
// The overall logic for DCA is as follows:
//  1. If both endpoint override and client certificate are specified, use them as is.
//  2. If user does not specify client certificate, we will attempt to use default
//     client certificate.
//  3. If user does not specify endpoint override, we will use defaultMtlsEndpoint if
//     client certificate is available and defaultEndpoint otherwise.
//
// Implications of the above logic:
//  1. If the user specifies a non-mTLS endpoint override but client certificate is
//     available, we will pass along the cert anyway and let the server decide what to do.
//  2. If the user specifies an mTLS endpoint override but client certificate is not
//     available, we will not fail-fast, but let backend throw error when connecting.
//
// If running within Google's cloud environment, and client certificate is not specified
// and not available through DCA, we will try mTLS with credentials held by
// the Secure Session Agent, which is part of Google's cloud infrastructure.
//
// We would like to avoid introducing client-side logic that parses whether the
// endpoint override is an mTLS url, since the url pattern may change at anytime.
//
// This package is not intended for use by end developers. Use the
// google.golang.org/api/option package to configure API clients.

// Package internal supports the options and transport packages.
package internal

import (
	"context"
	"crypto/tls"
	"net"
	"net/url"
	"os"
	"strings"

	"github.com/google/s2a-go"
	"github.com/google/s2a-go/fallback"
	"google.golang.org/api/internal/cert"
	"google.golang.org/grpc/credentials"
)

const (
	mTLSModeAlways = "always"
	mTLSModeNever  = "never"
	mTLSModeAuto   = "auto"

	// Experimental: if true, the code will try MTLS with S2A as the default for transport security. Default value is false.
	googleAPIUseS2AEnv = "EXPERIMENTAL_GOOGLE_API_USE_S2A"
)

// getClientCertificateSourceAndEndpoint is a convenience function that invokes
// getClientCertificateSource and getEndpoint sequentially and returns the client
// cert source and endpoint as a tuple.
func getClientCertificateSourceAndEndpoint(settings *DialSettings) (cert.Source, string, error) {
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

type transportConfig struct {
	clientCertSource cert.Source // The client certificate source.
	endpoint         string      // The corresponding endpoint to use based on client certificate source.
	s2aAddress       string      // The S2A address if it can be used, otherwise an empty string.
	s2aMTLSEndpoint  string      // The MTLS endpoint to use with S2A.
}

func getTransportConfig(settings *DialSettings) (*transportConfig, error) {
	clientCertSource, endpoint, err := getClientCertificateSourceAndEndpoint(settings)
	if err != nil {
		return &transportConfig{
			clientCertSource: nil, endpoint: "", s2aAddress: "", s2aMTLSEndpoint: "",
		}, err
	}
	defaultTransportConfig := transportConfig{
		clientCertSource: clientCertSource,
		endpoint:         endpoint,
		s2aAddress:       "",
		s2aMTLSEndpoint:  "",
	}

	if !shouldUseS2A(clientCertSource, settings) {
		return &defaultTransportConfig, nil
	}

	s2aMTLSEndpoint := settings.DefaultMTLSEndpoint
	// If there is endpoint override, honor it.
	if settings.Endpoint != "" {
		s2aMTLSEndpoint = endpoint
	}
	s2aAddress := GetS2AAddress()
	if s2aAddress == "" {
		return &defaultTransportConfig, nil
	}
	return &transportConfig{
		clientCertSource: clientCertSource,
		endpoint:         endpoint,
		s2aAddress:       s2aAddress,
		s2aMTLSEndpoint:  s2aMTLSEndpoint,
	}, nil
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
func getClientCertificateSource(settings *DialSettings) (cert.Source, error) {
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
func getEndpoint(settings *DialSettings, clientCertSource cert.Source) (string, error) {
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

// GetGRPCTransportConfigAndEndpoint returns an instance of credentials.TransportCredentials, and the
// corresponding endpoint to use for GRPC client.
func GetGRPCTransportConfigAndEndpoint(settings *DialSettings) (credentials.TransportCredentials, string, error) {
	config, err := getTransportConfig(settings)
	if err != nil {
		return nil, "", err
	}

	defaultTransportCreds := credentials.NewTLS(&tls.Config{
		GetClientCertificate: config.clientCertSource,
	})
	if config.s2aAddress == "" {
		return defaultTransportCreds, config.endpoint, nil
	}

	var fallbackOpts *s2a.FallbackOptions
	// In case of S2A failure, fall back to the endpoint that would've been used without S2A.
	if fallbackHandshake, err := fallback.DefaultFallbackClientHandshakeFunc(config.endpoint); err == nil {
		fallbackOpts = &s2a.FallbackOptions{
			FallbackClientHandshakeFunc: fallbackHandshake,
		}
	}

	s2aTransportCreds, err := s2a.NewClientCreds(&s2a.ClientOptions{
		S2AAddress:   config.s2aAddress,
		FallbackOpts: fallbackOpts,
	})
	if err != nil {
		// Use default if we cannot initialize S2A client transport credentials.
		return defaultTransportCreds, config.endpoint, nil
	}
	return s2aTransportCreds, config.s2aMTLSEndpoint, nil
}

// GetHTTPTransportConfigAndEndpoint returns a client certificate source, a function for dialing MTLS with S2A,
// and the endpoint to use for HTTP client.
func GetHTTPTransportConfigAndEndpoint(settings *DialSettings) (cert.Source, func(context.Context, string, string) (net.Conn, error), string, error) {
	config, err := getTransportConfig(settings)
	if err != nil {
		return nil, nil, "", err
	}

	if config.s2aAddress == "" {
		return config.clientCertSource, nil, config.endpoint, nil
	}

	var fallbackOpts *s2a.FallbackOptions
	// In case of S2A failure, fall back to the endpoint that would've been used without S2A.
	if fallbackURL, err := url.Parse(config.endpoint); err == nil {
		if fallbackDialer, fallbackServerAddr, err := fallback.DefaultFallbackDialerAndAddress(fallbackURL.Hostname()); err == nil {
			fallbackOpts = &s2a.FallbackOptions{
				FallbackDialer: &s2a.FallbackDialer{
					Dialer:     fallbackDialer,
					ServerAddr: fallbackServerAddr,
				},
			}
		}
	}

	dialTLSContextFunc := s2a.NewS2ADialTLSContextFunc(&s2a.ClientOptions{
		S2AAddress:   config.s2aAddress,
		FallbackOpts: fallbackOpts,
	})
	return nil, dialTLSContextFunc, config.s2aMTLSEndpoint, nil
}

func shouldUseS2A(clientCertSource cert.Source, settings *DialSettings) bool {
	// If client cert is found, use that over S2A.
	if clientCertSource != nil {
		return false
	}
	// If EXPERIMENTAL_GOOGLE_API_USE_S2A is not set to true, skip S2A.
	if !isGoogleS2AEnabled() {
		return false
	}
	// If DefaultMTLSEndpoint is not set and no endpoint override, skip S2A.
	if settings.DefaultMTLSEndpoint == "" && settings.Endpoint == "" {
		return false
	}
	// If MTLS is not enabled for this endpoint, skip S2A.
	if !mtlsEndpointEnabledForS2A() {
		return false
	}
	// If custom HTTP client is provided, skip S2A.
	if settings.HTTPClient != nil {
		return false
	}
	return true
}

// mtlsEndpointEnabledForS2A checks if the endpoint is indeed MTLS-enabled, so that we can use S2A for MTLS connection.
var mtlsEndpointEnabledForS2A = func() bool {
	// TODO(xmenxk): determine this via discovery config.
	return true
}

func isGoogleS2AEnabled() bool {
	return strings.ToLower(os.Getenv(googleAPIUseS2AEnv)) == "true"
}
