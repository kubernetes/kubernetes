package osincli

import (
	"errors"
	"net/http"
	"net/url"
)

// Caches configuration URLs
type clientconfigCache struct {
	authorizeUrl *url.URL
	tokenUrl     *url.URL
}

type Client struct {
	// caches urls
	configcache clientconfigCache

	// Client configuration
	config *ClientConfig

	// Transport is the HTTP transport to use when making requests.
	// It will default to http.DefaultTransport if nil.
	Transport http.RoundTripper
}

// Creates a new client
func NewClient(config *ClientConfig) (*Client, error) {
	c := &Client{
		config: config,
	}
	return c, c.initialize()
}

func (c *Client) initialize() error {
	if c.config.ClientId == "" || c.config.ClientSecret == "" || c.config.AuthorizeUrl == "" ||
		c.config.TokenUrl == "" || c.config.RedirectUrl == "" {
		return errors.New("Missing configuration")
	}

	var err error

	// cache configurations
	c.configcache.authorizeUrl, err = url.Parse(c.config.AuthorizeUrl)
	if err != nil {
		return err
	}

	c.configcache.tokenUrl, err = url.Parse(c.config.TokenUrl)
	if err != nil {
		return err
	}

	return nil
}
