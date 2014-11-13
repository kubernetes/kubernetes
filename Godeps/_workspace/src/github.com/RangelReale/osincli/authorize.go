package osincli

import (
	"errors"
	"net/http"
	"net/url"
)

type AuthorizeRequestType string

const (
	CODE AuthorizeRequestType = "code"
	//TOKEN                      = "token"	// token not supported in server applications (uses url fragment)
)

// Authorize request information
type AuthorizeRequest struct {
	client           *Client
	Type             AuthorizeRequestType
	CustomParameters map[string]string
}

// Authorization data
type AuthorizeData struct {
	Code  string
	State string
}

// Creates a new authorize request
func (c *Client) NewAuthorizeRequest(t AuthorizeRequestType) *AuthorizeRequest {
	return &AuthorizeRequest{
		client:           c,
		Type:             t,
		CustomParameters: make(map[string]string),
	}
}

// Returns the authorize url
func (c *AuthorizeRequest) GetAuthorizeUrl() *url.URL {
	return c.GetAuthorizeUrlWithParams("")
}

// Returns the authorize url
func (c *AuthorizeRequest) GetAuthorizeUrlWithParams(state string) *url.URL {
	u := *c.client.configcache.authorizeUrl
	uq := u.Query()
	uq.Add("response_type", string(c.Type))
	uq.Add("client_id", c.client.config.ClientId)
	uq.Add("redirect_uri", c.client.config.RedirectUrl)
	if c.client.config.Scope != "" {
		uq.Add("scope", c.client.config.Scope)
	}
	if state != "" {
		uq.Add("state", state)
	}
	if c.CustomParameters != nil {
		for pn, pv := range c.CustomParameters {
			uq.Add(pn, pv)
		}
	}
	u.RawQuery = uq.Encode()
	return &u
}

// Handle the authorization request
func (c *AuthorizeRequest) HandleRequest(r *http.Request) (*AuthorizeData, error) {
	r.ParseForm()

	var ad *AuthorizeData

	if c.Type == CODE {
		// detect error parameters
		if r.Form.Get("error") != "" {
			return nil, NewError(r.Form.Get("error"), r.Form.Get("error_description"), r.Form.Get("error_uri"), r.Form.Get("state"))
		} else if r.Form.Get("code") == "" {
			return nil, errors.New("Requested parameter not sent")
		}
		ad = &AuthorizeData{
			Code:  r.Form.Get("code"),
			State: r.Form.Get("state"),
		}
	} else {
		return nil, errors.New("Unsupported response type")
	}

	return ad, nil
}
