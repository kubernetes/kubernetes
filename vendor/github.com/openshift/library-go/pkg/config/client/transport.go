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

func (t *preferredHostRT) RoundTrip(r *http.Request) (*http.Response, error) {
	preferredHost := t.preferredHostFn()

	if len(preferredHost) == 0 {
		return t.baseRT.RoundTrip(r)
	}

	r.Host = preferredHost
	r.URL.Host = preferredHost
	return t.baseRT.RoundTrip(r)
}
