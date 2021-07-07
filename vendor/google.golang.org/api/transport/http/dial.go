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
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"go.opencensus.io/plugin/ochttp"
	"golang.org/x/oauth2"
	"google.golang.org/api/googleapi/transport"
	"google.golang.org/api/internal"
	"google.golang.org/api/option"
	"google.golang.org/api/transport/cert"
	"google.golang.org/api/transport/http/internal/propagation"
)

const (
	mTLSModeAlways = "always"
	mTLSModeNever  = "never"
	mTLSModeAuto   = "auto"
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

		ts := creds.TokenSource
		if settings.TokenSource != nil {
			ts = settings.TokenSource
		}
		trans = &oauth2.Transport{
			Base:   trans,
			Source: ts,
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
// Otherwise, use a default transport, taking most defaults from
// http.DefaultTransport.
// If TLSCertificate is available, set TLSClientConfig as well.
func defaultBaseTransport(ctx context.Context, clientCertSource cert.Source) http.RoundTripper {
	if appengineUrlfetchHook != nil {
		return appengineUrlfetchHook(ctx)
	}

	// Copy http.DefaultTransport except for MaxIdleConnsPerHost setting,
	// which is increased due to reported performance issues under load in the GCS
	// client. Transport.Clone is only available in Go 1.13 and up.
	trans := clonedTransport(http.DefaultTransport)
	if trans == nil {
		trans = fallbackBaseTransport()
	}
	trans.MaxIdleConnsPerHost = 100

	if clientCertSource != nil {
		trans.TLSClientConfig = &tls.Config{
			GetClientCertificate: clientCertSource,
		}
	}

	return trans
}

// fallbackBaseTransport is used in <go1.13 as well as in the rare case if
// http.DefaultTransport has been reassigned something that's not a
// *http.Transport.
func fallbackBaseTransport() *http.Transport {
	return &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
			DualStack: true,
		}).DialContext,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}
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
	if settings.HTTPClient != nil {
		return nil, nil // HTTPClient is incompatible with ClientCertificateSource
	} else if settings.ClientCertSource != nil {
		return settings.ClientCertSource, nil
	} else {
		return cert.DefaultSource()
	}

}

// getEndpoint returns the endpoint for the service, taking into account the
// user-provided endpoint override "settings.Endpoint"
//
// If no endpoint override is specified, we will either return the default endpoint or
// the default mTLS endpoint if a client certificate is available.
//
// You can override the default endpoint (mtls vs. regular) by setting the
// GOOGLE_API_USE_MTLS environment variable.
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
		return "", errors.New("WithEndpoint requires a full URL path")
	}

	// Assume user-provided endpoint is host[:port], merge it with the default endpoint.
	return mergeEndpoints(settings.DefaultEndpoint, settings.Endpoint)
}

func getMTLSMode() string {
	mode := os.Getenv("GOOGLE_API_USE_MTLS")
	if mode == "" {
		// TODO(shinfan): Update this to "auto" when the mTLS feature is fully released.
		return mTLSModeNever
	}
	return strings.ToLower(mode)
}

func mergeEndpoints(base, newHost string) (string, error) {
	u, err := url.Parse(base)
	if err != nil {
		return "", err
	}
	u.Host = newHost
	return u.String(), nil
}
