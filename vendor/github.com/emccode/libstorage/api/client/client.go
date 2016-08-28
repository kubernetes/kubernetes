package client

import (
	"net/http"

	"github.com/emccode/libstorage/api/context"
	"github.com/emccode/libstorage/api/types"
)

func init() {
	context.RegisterCustomKey(transactionHeaderKey, context.CustomHeaderKey)
	context.RegisterCustomKey(instanceIDHeaderKey, context.CustomHeaderKey)
	context.RegisterCustomKey(localDevicesHeaderKey, context.CustomHeaderKey)
}

// Client is the libStorage API client.
type client struct {
	http.Client
	host         string
	logRequests  bool
	logResponses bool
	serverName   string
}

// New returns a new API client.
func New(host string, transport *http.Transport) types.APIClient {
	return &client{
		Client: http.Client{
			Transport: transport,
		},
		host: host,
	}
}

func (c *client) ServerName() string {
	return c.serverName
}

func (c *client) LogRequests(enabled bool) {
	c.logRequests = enabled
}

func (c *client) LogResponses(enabled bool) {
	c.logResponses = enabled
}
