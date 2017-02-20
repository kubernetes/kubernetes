package gophercloud

import (
	"io"
	"net/http"
	"strings"
)

// ServiceClient stores details required to interact with a specific service API implemented by a provider.
// Generally, you'll acquire these by calling the appropriate `New` method on a ProviderClient.
type ServiceClient struct {
	// ProviderClient is a reference to the provider that implements this service.
	*ProviderClient

	// Endpoint is the base URL of the service's API, acquired from a service catalog.
	// It MUST end with a /.
	Endpoint string

	// ResourceBase is the base URL shared by the resources within a service's API. It should include
	// the API version and, like Endpoint, MUST end with a / if set. If not set, the Endpoint is used
	// as-is, instead.
	ResourceBase string

	Microversion string
}

// ResourceBaseURL returns the base URL of any resources used by this service. It MUST end with a /.
func (client *ServiceClient) ResourceBaseURL() string {
	if client.ResourceBase != "" {
		return client.ResourceBase
	}
	return client.Endpoint
}

// ServiceURL constructs a URL for a resource belonging to this provider.
func (client *ServiceClient) ServiceURL(parts ...string) string {
	return client.ResourceBaseURL() + strings.Join(parts, "/")
}

// Get calls `Request` with the "GET" HTTP verb.
func (client *ServiceClient) Get(url string, JSONResponse interface{}, opts *RequestOpts) (*http.Response, error) {
	if opts == nil {
		opts = &RequestOpts{}
	}
	if JSONResponse != nil {
		opts.JSONResponse = JSONResponse
	}

	if opts.MoreHeaders == nil {
		opts.MoreHeaders = make(map[string]string)
	}
	opts.MoreHeaders["X-OpenStack-Nova-API-Version"] = client.Microversion

	return client.Request("GET", url, opts)
}

// Post calls `Request` with the "POST" HTTP verb.
func (client *ServiceClient) Post(url string, JSONBody interface{}, JSONResponse interface{}, opts *RequestOpts) (*http.Response, error) {
	if opts == nil {
		opts = &RequestOpts{}
	}

	if v, ok := (JSONBody).(io.Reader); ok {
		opts.RawBody = v
	} else if JSONBody != nil {
		opts.JSONBody = JSONBody
	}

	if JSONResponse != nil {
		opts.JSONResponse = JSONResponse
	}

	if opts.MoreHeaders == nil {
		opts.MoreHeaders = make(map[string]string)
	}
	opts.MoreHeaders["X-OpenStack-Nova-API-Version"] = client.Microversion

	return client.Request("POST", url, opts)
}

// Put calls `Request` with the "PUT" HTTP verb.
func (client *ServiceClient) Put(url string, JSONBody interface{}, JSONResponse interface{}, opts *RequestOpts) (*http.Response, error) {
	if opts == nil {
		opts = &RequestOpts{}
	}

	if v, ok := (JSONBody).(io.Reader); ok {
		opts.RawBody = v
	} else if JSONBody != nil {
		opts.JSONBody = JSONBody
	}

	if JSONResponse != nil {
		opts.JSONResponse = JSONResponse
	}

	if opts.MoreHeaders == nil {
		opts.MoreHeaders = make(map[string]string)
	}
	opts.MoreHeaders["X-OpenStack-Nova-API-Version"] = client.Microversion

	return client.Request("PUT", url, opts)
}

// Patch calls `Request` with the "PATCH" HTTP verb.
func (client *ServiceClient) Patch(url string, JSONBody interface{}, JSONResponse interface{}, opts *RequestOpts) (*http.Response, error) {
	if opts == nil {
		opts = &RequestOpts{}
	}

	if v, ok := (JSONBody).(io.Reader); ok {
		opts.RawBody = v
	} else if JSONBody != nil {
		opts.JSONBody = JSONBody
	}

	if JSONResponse != nil {
		opts.JSONResponse = JSONResponse
	}

	if opts.MoreHeaders == nil {
		opts.MoreHeaders = make(map[string]string)
	}
	opts.MoreHeaders["X-OpenStack-Nova-API-Version"] = client.Microversion

	return client.Request("PATCH", url, opts)
}

// Delete calls `Request` with the "DELETE" HTTP verb.
func (client *ServiceClient) Delete(url string, opts *RequestOpts) (*http.Response, error) {
	if opts == nil {
		opts = &RequestOpts{}
	}

	if opts.MoreHeaders == nil {
		opts.MoreHeaders = make(map[string]string)
	}
	opts.MoreHeaders["X-OpenStack-Nova-API-Version"] = client.Microversion

	return client.Request("DELETE", url, opts)
}
