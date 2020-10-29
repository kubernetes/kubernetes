package paramtoken

import (
	"net/http"
	"strings"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/util/wsstream"
)

// Authenticator provides a way to authenticate tokens provided as a parameter
// This only exists to allow websocket connections to use an API token, since they cannot set an Authorize header
// For this authenticator to work, tokens will be part of the request URL, and are more likely to be logged or otherwise exposed.
// Every effort should be made to filter tokens from being logged when using this authenticator.
type Authenticator struct {
	// param is the query param to use as a token
	param string
	// auth is the token authenticator to use to validate the token
	auth authenticator.Token
	// removeParam indicates whether the parameter should be stripped from the incoming request
	removeParam bool
}

func New(param string, auth authenticator.Token, removeParam bool) *Authenticator {
	return &Authenticator{param, auth, removeParam}
}

func (a *Authenticator) AuthenticateRequest(req *http.Request) (*authenticator.Response, bool, error) {
	// Only accept query param auth for websocket connections
	if !wsstream.IsWebSocketRequest(req) {
		return nil, false, nil
	}

	q := req.URL.Query()
	token := strings.TrimSpace(q.Get(a.param))
	if token == "" {
		return nil, false, nil
	}
	authResponse, ok, err := a.auth.AuthenticateToken(req.Context(), token)
	if ok && a.removeParam {
		q.Del(a.param)
		req.URL.RawQuery = q.Encode()
	}
	return authResponse, ok, err
}
