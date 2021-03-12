// Copyright 2015 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package http supports network connections to HTTP servers.
// This package is not intended for use by end developers. Use the
// google.golang.org/api/option package to configure API clients.
package http

import (
	"context"
	"crypto/tls"
	"errors"
	"net/http"
	"net/url"
	"strings"

	"go.opencensus.io/plugin/ochttp"
	"golang.org/x/oauth2"
	"google.golang.org/api/googleapi/transport"
	"google.golang.org/api/internal"
	"google.golang.org/api/option"
	"google.golang.org/api/transport/cert"
	"google.golang.org/api/transport/http/internal/propagation"
)

// NewClient returns an HTTP client for use communicating with a Google cloud
// service, configured with the given ClientOptions. It also returns the endpoint
// for the service as specified in the options.
func NewClient(ctx context.Context, opts ...option.ClientOption) (*http.Client, string, error) {
	settings, err := newSettings(opts)
	if err != nil {
		return nil, "", err
	}
	clientCertSource, err := getClientCertificateSource(settings)
	if err != nil {
		return nil, "", err
	}
	endpoint, err := getEndpoint(settings, clientCertSource)
	if err != nil {
		return nil, "", err
	}
	// TODO(cbro): consider injecting the User-Agent even if an explicit HTTP client is provided?
	if settings.HTTPClient != nil {
		return settings.HTTPClient, endpoint, nil
	}
	trans, err := newTransport(ctx, defaultBaseTransport(ctx, clientCertSource), settings)
	if err != nil {
		return nil, "", err
	}
	return &http.Client{Transport: trans}, endpoint, nil
}

// NewTransport creates an http.RoundTripper for use communicating with a Google
// cloud service, configured with the given ClientOptions. Its RoundTrip method delegates to base.
func NewTransport(ctx context.Context, base http.RoundTripper, opts ...option.ClientOption) (http.RoundTripper, error) {
	settings, err := newSettings(opts)
	if err != nil {
		return nil, err
	}
	if settings.HTTPClient != nil {
		return nil, errors.New("transport/http: WithHTTPClient passed to NewTransport")
	}
	return newTransport(ctx, base, settings)
}

func newTransport(ctx context.Context, base http.RoundTripper, settings *internal.DialSettings) (http.RoundTripper, error) {
	paramTransport := &parameterTransport{
		base:          base,
		userAgent:     settings.UserAgent,
		quotaProject:  settings.QuotaProject,
		requestReason: settings.RequestReason,
	}
	var trans http.RoundTripper = paramTransport
	trans = addOCTransport(trans, settings)
	switch {
	case settings.NoAuth:
		// Do nothing.
	case settings.APIKey != "":
		trans = &transport.APIKey{
			Transport: trans,
			Key:       settings.APIKey,
		}
	default:
		creds, err := internal.Creds(ctx, settings)
		if err != nil {
			return nil, err
		}
		if paramTransport.quotaProject == "" {
			paramTransport.quotaProject = internal.QuotaProjectFromCreds(creds)
		}
		trans = &oauth2.Transport{
			Base:   trans,
			Source: creds.TokenSource,
		}
	}
	return trans, nil
}

func newSettings(opts []option.ClientOption) (*internal.DialSettings, error) {
	var o internal.DialSettings
	for _, opt := range opts {
		opt.Apply(&o)
	}
	if err := o.Validate(); err != nil {
		return nil, err
	}
	if o.GRPCConn != nil {
		return nil, errors.New("unsupported gRPC connection specified")
	}
	return &o, nil
}

type parameterTransport struct {
	userAgent     string
	quotaProject  string
	requestReason string

	base http.RoundTripper
}

func (t *parameterTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	rt := t.base
	if rt == nil {
		return nil, errors.New("transport: no Transport specified")
	}
	newReq := *req
	newReq.Header = make(http.Header)
	for k, vv := range req.Header {
		newReq.Header[k] = vv
	}
	if t.userAgent != "" {
		// TODO(cbro): append to existing User-Agent header?
		newReq.Header.Set("User-Agent", t.userAgent)
	}

	// Attach system parameters into the header
	if t.quotaProject != "" {
		newReq.Header.Set("X-Goog-User-Project", t.quotaProject)
	}
	if t.requestReason != "" {
		newReq.Header.Set("X-Goog-Request-Reason", t.requestReason)
	}

	return rt.RoundTrip(&newReq)
}

// Set at init time by dial_appengine.go. If nil, we're not on App Engine.
var appengineUrlfetchHook func(context.Context) http.RoundTripper

// defaultBaseTransport returns the base HTTP transport.
// On App Engine, this is urlfetch.Transport.
// If TLSCertificate is available, return a custom Transport with TLSClientConfig.
// Otherwise, return http.DefaultTransport.
func defaultBaseTransport(ctx context.Context, clientCertSource cert.Source) http.RoundTripper {
	if appengineUrlfetchHook != nil {
		return appengineUrlfetchHook(ctx)
	}

	if clientCertSource != nil {
		// TODO (cbro): copy default transport settings from http.DefaultTransport
		return &http.Transport{
			TLSClientConfig: &tls.Config{
				GetClientCertificate: clientCertSource,
			},
		}
	}

	return http.DefaultTransport
}

func addOCTransport(trans http.RoundTripper, settings *internal.DialSettings) http.RoundTripper {
	if settings.TelemetryDisabled {
		return trans
	}
	return &ochttp.Transport{
		Base:        trans,
		Propagation: &propagation.HTTPFormat{},
	}
}

// getClientCertificateSource returns a default client certificate source, if
// not provided by the user.
//
// A nil default source can be returned if the source does not exist. Any exceptions
// encountered while initializing the default source will be reported as client
// error (ex. corrupt metadata file).
//
// The overall logic is as follows:
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
func getClientCertificateSource(settings *internal.DialSettings) (cert.Source, error) {
	if settings.ClientCertSource != nil {
		return settings.ClientCertSource, nil
	}
	return cert.DefaultSource()
}

// getEndpoint returns the endpoint for the service, taking into account the
// user-provided endpoint override "settings.Endpoint"
//
// If no endpoint override is specified, we will return the default endpoint (or
// the default mTLS endpoint if a client certificate is available).
//
// If the endpoint override is an address (host:port) rather than full base
// URL (ex. https://...), then the user-provided address will be merged into
// the default endpoint. For example, WithEndpoint("myhost:8000") and
// WithDefaultEndpoint("https://foo.com/bar/baz") will return "https://myhost:8080/bar/baz"
func getEndpoint(settings *internal.DialSettings, clientCertSource cert.Source) (string, error) {
	if settings.Endpoint == "" {
		if clientCertSource != nil {
			return generateDefaultMtlsEndpoint(settings.DefaultEndpoint), nil
		}
		return settings.DefaultEndpoint, nil
	}
	if strings.Contains(settings.Endpoint, "://") {
		// User passed in a full URL path, use it verbatim.
		return settings.Endpoint, nil
	}
	if settings.DefaultEndpoint == "" {
		return "", errors.New("WithEndpoint requires a full URL path")
	}

	// Assume user-provided endpoint is host[:port], merge it with the default endpoint.
	return mergeEndpoints(settings.DefaultEndpoint, settings.Endpoint)
}

func mergeEndpoints(base, newHost string) (string, error) {
	u, err := url.Parse(base)
	if err != nil {
		return "", err
	}
	u.Host = newHost
	return u.String(), nil
}

// generateDefaultMtlsEndpoint attempts to derive the mTLS version of the
// defaultEndpoint via regex, and returns defaultEndpoint if unsuccessful.
//
// We need to applying the following 2 transformations:
// 1. pubsub.googleapis.com to pubsub.mtls.googleapis.com
// 2. pubsub.sandbox.googleapis.com to pubsub.mtls.sandbox.googleapis.com
//
// TODO(andyzhao): In the future, the mTLS endpoint will be read from the Discovery Document
// and passed in as defaultMtlsEndpoint instead of generated from defaultEndpoint,
// and this function will be removed.
func generateDefaultMtlsEndpoint(defaultEndpoint string) string {
	var domains = []string{
		".sandbox.googleapis.com", // must come first because .googleapis.com is a substring
		".googleapis.com",
	}
	for _, domain := range domains {
		if strings.Contains(defaultEndpoint, domain) {
			return strings.Replace(defaultEndpoint, domain, ".mtls"+domain, -1)
		}
	}
	return defaultEndpoint
}
