package client

import (
	"net"
	"net/http"
	"net/url"

	"k8s.io/client-go/rest"
)

// AnonymousClientConfigWithWrapTransport returns a copy of the given config with all user credentials (cert/key, bearer token, and username/password) and custom transports (Transport) removed.
// This function preserves WrapTransport for clients that care about custom HTTP behavior.
func AnonymousClientConfigWithWrapTransport(config *rest.Config) *rest.Config {
	newConfig := rest.AnonymousClientConfig(config)
	newConfig.WrapTransport = config.WrapTransport
	return newConfig
}

// DefaultServerName extract the hostname from the config.Host and sets it in config.ServerName
// the ServerName is passed to the server for SNI and is used in the client to check server certificates.
//
// note:
// if the ServerName has been already specified calling this method has no effect
func DefaultServerName(config *rest.Config) error {
	if len(config.ServerName) > 0 {
		return nil
	}
	u, err := url.Parse(config.Host)
	if err != nil {
		return err
	}
	host, _, err := net.SplitHostPort(u.Host)
	if err != nil {
		// assume u.Host contains only host portion
		config.ServerName = u.Host
		return nil
	}
	config.ServerName = host
	return nil
}

// NewPreferredHostRoundTripper a simple middleware for changing the destination host for each request to the provided one.
// If the preferred host doesn't exists (an empty string) then this RT has no effect.
func NewPreferredHostRoundTripper(preferredHostFn func() string) func(http.RoundTripper) http.RoundTripper {
	return func(rt http.RoundTripper) http.RoundTripper {
		return &preferredHostRT{baseRT: rt, preferredHostFn: preferredHostFn}
	}
}

type preferredHostRT struct {
	baseRT          http.RoundTripper
	preferredHostFn func() string
}

func (rt *preferredHostRT) RoundTrip(r *http.Request) (*http.Response, error) {
	preferredHost := rt.preferredHostFn()

	if len(preferredHost) == 0 {
		return rt.baseRT.RoundTrip(r)
	}

	r.Host = preferredHost
	r.URL.Host = preferredHost
	return rt.baseRT.RoundTrip(r)
}

// CancelRequest exists to facilitate cancellation.
//
// In general there are at least three ways of cancelling a request by an HTTP client:
// 1. Transport.CancelRequest (depreciated)
// 2. Request.Cancel
// 3. Request.Context (preferred)
//
// While using client-go callers can specify a timeout value that gets passed directly to an http.Client.
// The HTTP client cancels requests to the underlying Transport as if the Request's Context ended.
// For compatibility, the Client will also use the deprecated CancelRequest method on Transport if found.
// New RoundTripper implementations should use the Request's Context for cancellation instead of implementing CancelRequest.
//
// Because this wrapper might be the first or might be actually wrapped with already existing wrappers that already implement CancelRequest we need to simply conform.
//
// See for more details:
//
//	https://github.com/kubernetes/kubernetes/blob/442a69c3bdf6fe8e525b05887e57d89db1e2f3a5/staging/src/k8s.io/client-go/transport/transport.go#L257
//	https://github.com/kubernetes/kubernetes/blob/e29c568c4a9cd45d15665345aa015e21bcff52dd/staging/src/k8s.io/client-go/rest/config.go#L328
//	https://github.com/kubernetes/kubernetes/blob/3b2746c9ea9e0fa247b01dca27634e509b385eda/staging/src/k8s.io/client-go/transport/round_trippers.go#L302
func (rt *preferredHostRT) CancelRequest(req *http.Request) {
	type canceler interface{ CancelRequest(*http.Request) }

	if rtCanceller, ok := rt.baseRT.(canceler); ok {
		rtCanceller.CancelRequest(req)
	}
}
