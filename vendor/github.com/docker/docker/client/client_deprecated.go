package client

import "net/http"

// NewClient initializes a new API client for the given host and API version.
// It uses the given http client as transport.
// It also initializes the custom http headers to add to each request.
//
// It won't send any version information if the version number is empty. It is
// highly recommended that you set a version or your client may break if the
// server is upgraded.
// Deprecated: use NewClientWithOpts
func NewClient(host string, version string, client *http.Client, httpHeaders map[string]string) (*Client, error) {
	return NewClientWithOpts(WithHost(host), WithVersion(version), WithHTTPClient(client), WithHTTPHeaders(httpHeaders))
}

// NewEnvClient initializes a new API client based on environment variables.
// See FromEnv for a list of support environment variables.
//
// Deprecated: use NewClientWithOpts(FromEnv)
func NewEnvClient() (*Client, error) {
	return NewClientWithOpts(FromEnv)
}
