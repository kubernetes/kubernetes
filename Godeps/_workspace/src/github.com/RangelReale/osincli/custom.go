package osincli

import (
	"net/url"
)

// Custom request information
type CustomRequest struct {
	client           *Client
	CustomParameters map[string]string
}

// Custom data
type CustomData struct {
	ResponseData ResponseData
}

func (c *Client) NewCustomRequest() *CustomRequest {
	return &CustomRequest{
		client: c,
	}
}

// Send a custom request
// On OAuth2 error, and osincli.Error is returned as error
func (c *CustomRequest) GetRequest(url *url.URL) (*CustomData, error) {
	var ba *BasicAuth
	if !c.client.config.SendClientSecretInParams {
		ba = &BasicAuth{Username: c.client.config.ClientId, Password: c.client.config.ClientSecret}
	}

	// return value
	ret := &CustomData{
		ResponseData: make(ResponseData),
	}

	// download data
	m := "POST"
	if c.client.config.UseGetAccessRequest {
		m = "GET"
	}
	err := downloadData(m, url, ba, c.client.Transport, ret.ResponseData)
	if err != nil {
		return nil, err
	}

	return ret, nil
}
