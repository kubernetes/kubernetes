package gophercloud

import (
	"net/http"
	"strings"
	"fmt"
	"github.com/tonnerre/golang-pretty"
)

// Provider structures exist for each tangible provider of OpenStack service.
// For example, Rackspace, Hewlett-Packard, and NASA might have their own instance of this structure.
//
// At a minimum, a provider must expose an authentication endpoint.
type Provider struct {
	AuthEndpoint string
}

// ReauthHandlerFunc functions are responsible for somehow performing the task of
// reauthentication.
type ReauthHandlerFunc func(AccessProvider) error

// Context structures encapsulate Gophercloud-global state in a manner which
// facilitates easier unit testing.  As a user of this SDK, you'll never
// have to use this structure, except when contributing new code to the SDK.
type Context struct {
	// providerMap serves as a directory of supported providers.
	providerMap map[string]Provider

	// httpClient refers to the current HTTP client interface to use.
	httpClient *http.Client

	// reauthHandler provides the functionality needed to re-authenticate
	// if that feature is enabled.  Note: in order to allow for automatic
	// re-authentication, the Context object will need to remember your
	// username, password, and tenant ID as provided in the initial call
	// to Authenticate().  If you do not desire this, you'll need to handle
	// reauthentication yourself through other means.  Two methods exist:
	// the first approach is to just handle errors yourself at the application
	// layer, and the other is through a custom reauthentication handler
	// set through the WithReauthHandler() method.
	reauthHandler ReauthHandlerFunc
}

// TestContext yields a new Context instance, pre-initialized with a barren
// state suitable for per-unit-test customization.  This configuration consists
// of:
//
// * An empty provider map.
//
// * An HTTP client built by the net/http package (see http://godoc.org/net/http#Client).
func TestContext() *Context {
	return &Context{
		providerMap: make(map[string]Provider),
		httpClient:  &http.Client{},
		reauthHandler: func(acc AccessProvider) error {
			return acc.Reauthenticate()
		},
	}
}

// UseCustomClient configures the context to use a customized HTTP client
// instance.  By default, TestContext() will return a Context which uses
// the net/http package's default client instance.
func (c *Context) UseCustomClient(hc *http.Client) *Context {
	c.httpClient = hc
	return c
}

// RegisterProvider allows a unit test to register a mythical provider convenient for testing.
// If the provider structure lacks adequate configuration, or the configuration given has some
// detectable error, an ErrConfiguration error will result.
func (c *Context) RegisterProvider(name string, p Provider) error {
	if p.AuthEndpoint == "" {
		return ErrConfiguration
	}

	c.providerMap[name] = p
	return nil
}

// WithProvider offers convenience for unit tests.
func (c *Context) WithProvider(name string, p Provider) *Context {
	err := c.RegisterProvider(name, p)
	if err != nil {
		panic(err)
	}
	return c
}

// ProviderByName will locate a provider amongst those previously registered, if it exists.
// If the named provider has not been registered, an ErrProvider error will result.
//
// You may also specify a custom Identity API URL.
// Any provider name that contains the characters "://", in that order, will be treated as a custom Identity API URL.
// Custom URLs, important for private cloud deployments, overrides all provider configurations.
func (c *Context) ProviderByName(name string) (p Provider, err error) {
	for provider, descriptor := range c.providerMap {
		if name == provider {
			return descriptor, nil
		}
	}
	if strings.Contains(name, "://") {
		p = Provider{
			AuthEndpoint: name,
		}
		return p, nil
	}
	return Provider{}, ErrProvider
}

func getServiceCatalogFromAccessProvider(provider AccessProvider) ([]CatalogEntry) {
	access, found := provider.(*Access)
	if found {
		return access.ServiceCatalog
	} else {
		return nil
	}
}

// Instantiates a Cloud Servers API for the provider given.
func (c *Context) ServersApi(provider AccessProvider, criteria ApiCriteria) (CloudServersProvider, error) {
	url := provider.FirstEndpointUrlByCriteria(criteria)
	if url == "" {
		var err = fmt.Errorf(
			"Missing endpoint, or insufficient privileges to access endpoint; criteria = %# v; serviceCatalog = %# v",
			pretty.Formatter(criteria),
			pretty.Formatter(getServiceCatalogFromAccessProvider(provider)))
		return nil, err
	}

	gcp := &genericServersProvider{
		endpoint: url,
		context:  c,
		access:   provider,
	}

	return gcp, nil
}

// WithReauthHandler configures the context to handle reauthentication attempts using the supplied
// funtion.  By default, reauthentication happens by invoking Authenticate(), which is unlikely to be
// useful in a unit test.
//
// Do not confuse this function with WithReauth()!  Although they work together to support reauthentication,
// WithReauth() actually contains the decision-making logic to determine when to perform a reauth,
// while WithReauthHandler() is used to configure what a reauth actually entails.
func (c *Context) WithReauthHandler(f ReauthHandlerFunc) *Context {
	c.reauthHandler = f
	return c
}
