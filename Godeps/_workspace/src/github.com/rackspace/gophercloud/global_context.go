package gophercloud

import (
	"github.com/racker/perigee"
)

// globalContext is the, well, "global context."
// Most of this SDK is written in a manner to facilitate easier testing,
// which doesn't require all the configuration a real-world application would require.
// However, for real-world deployments, applications should be able to rely on a consistent configuration of providers, etc.
var globalContext *Context

// providers is the set of supported providers.
var providers = map[string]Provider{
	"rackspace-us": {
		AuthEndpoint: "https://identity.api.rackspacecloud.com/v2.0/tokens",
	},
	"rackspace-uk": {
		AuthEndpoint: "https://lon.identity.api.rackspacecloud.com/v2.0/tokens",
	},
}

// Initialize the global context to sane configuration.
// The Go runtime ensures this function is called before main(),
// thus guaranteeing proper configuration before your application ever runs.
func init() {
	globalContext = TestContext()
	for name, descriptor := range providers {
		globalContext.RegisterProvider(name, descriptor)
	}
}

// Authenticate() grants access to the OpenStack-compatible provider API.
//
// Providers are identified through a unique key string.
// Specifying an unsupported provider will result in an ErrProvider error.
// However, you may also specify a custom Identity API URL.
// Any provider name that contains the characters "://", in that order, will be treated as a custom Identity API URL.
// Custom URLs, important for private cloud deployments, overrides all provider configurations.
//
// The supplied AuthOptions instance allows the client to specify only those credentials
// relevant for the authentication request.  At present, support exists for OpenStack
// Identity V2 API only; support for V3 will become available as soon as documentation for it
// becomes readily available.
//
// For Identity V2 API requirements, you must provide at least the Username and Password
// options.  The TenantId field is optional, and defaults to "".
func Authenticate(provider string, options AuthOptions) (*Access, error) {
	return globalContext.Authenticate(provider, options)
}

// Instantiates a Cloud Servers object for the provider given.
func ServersApi(acc AccessProvider, criteria ApiCriteria) (CloudServersProvider, error) {
	return globalContext.ServersApi(acc, criteria)
}

// ActualResponseCode inspects a returned error, and discovers the actual response actual
// response code that caused the error to be raised.
func ActualResponseCode(e error) (int, error) {
	if err, typeOk := e.(*perigee.UnexpectedResponseCodeError); typeOk {
		return err.Actual, nil
	} else if err, typeOk := e.(*AuthError); typeOk{
		return err.StatusCode, nil
	}

	return 0, ErrError
}
