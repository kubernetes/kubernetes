package osin

import (
	"net/http"
	"net/url"
	"time"
)

// AuthorizeRequestType is the type for OAuth param `response_type`
type AuthorizeRequestType string

const (
	CODE  AuthorizeRequestType = "code"
	TOKEN                      = "token"
)

// Authorize request information
type AuthorizeRequest struct {
	Type        AuthorizeRequestType
	Client      Client
	Scope       string
	RedirectUri string
	State       string

	// Set if request is authorized
	Authorized bool

	// Token expiration in seconds. Change if different from default.
	// If type = TOKEN, this expiration will be for the ACCESS token.
	Expiration int32

	// Data to be passed to storage. Not used by the library.
	UserData interface{}

	// HttpRequest *http.Request for special use
	HttpRequest *http.Request
}

// Authorization data
type AuthorizeData struct {
	// Client information
	Client Client

	// Authorization code
	Code string

	// Token expiration in seconds
	ExpiresIn int32

	// Requested scope
	Scope string

	// Redirect Uri from request
	RedirectUri string

	// State data from request
	State string

	// Date created
	CreatedAt time.Time

	// Data to be passed to storage. Not used by the library.
	UserData interface{}
}

// IsExpired is true if authorization expired
func (d *AuthorizeData) IsExpired() bool {
	return d.CreatedAt.Add(time.Duration(d.ExpiresIn) * time.Second).Before(time.Now())
}

// ExpireAt returns the expiration date
func (d *AuthorizeData) ExpireAt() time.Time {
	return d.CreatedAt.Add(time.Duration(d.ExpiresIn) * time.Second)
}

// AuthorizeTokenGen is the token generator interface
type AuthorizeTokenGen interface {
	GenerateAuthorizeToken(data *AuthorizeData) (string, error)
}

// HandleAuthorizeRequest is the main http.HandlerFunc for handling
// authorization requests
func (s *Server) HandleAuthorizeRequest(w *Response, r *http.Request) *AuthorizeRequest {
	r.ParseForm()

	requestType := AuthorizeRequestType(r.Form.Get("response_type"))
	if s.Config.AllowedAuthorizeTypes.Exists(requestType) {
		switch requestType {
		case CODE:
			return s.handleCodeRequest(w, r)
		case TOKEN:
			return s.handleTokenRequest(w, r)
		}
	}

	w.SetError(E_UNSUPPORTED_RESPONSE_TYPE, "")
	return nil
}

func (s *Server) handleCodeRequest(w *Response, r *http.Request) *AuthorizeRequest {
	// create the authorization request
	unescapedUri, err := url.QueryUnescape(r.Form.Get("redirect_uri"))
	if err != nil {
		unescapedUri = ""
	}
	ret := &AuthorizeRequest{
		Type:        CODE,
		State:       r.Form.Get("state"),
		Scope:       r.Form.Get("scope"),
		RedirectUri: unescapedUri,
		Authorized:  false,
		Expiration:  s.Config.AuthorizationExpiration,
		HttpRequest: r,
	}

	// must have a valid client
	ret.Client, err = w.Storage.GetClient(r.Form.Get("client_id"))
	if err != nil {
		w.SetErrorState(E_SERVER_ERROR, "", ret.State)
		w.InternalError = err
		return nil
	}
	if ret.Client == nil {
		w.SetErrorState(E_UNAUTHORIZED_CLIENT, "", ret.State)
		return nil
	}
	if ret.Client.GetRedirectUri() == "" {
		w.SetErrorState(E_UNAUTHORIZED_CLIENT, "", ret.State)
		return nil
	}

	// force redirect response to client redirecturl first
	w.SetRedirect(FirstUri(ret.Client.GetRedirectUri(), s.Config.RedirectUriSeparator))

	// check redirect uri
	if ret.RedirectUri == "" {
		ret.RedirectUri = FirstUri(ret.Client.GetRedirectUri(), s.Config.RedirectUriSeparator)
	}
	if err = ValidateUriList(ret.Client.GetRedirectUri(), ret.RedirectUri, s.Config.RedirectUriSeparator); err != nil {
		w.SetErrorState(E_INVALID_REQUEST, "", ret.State)
		w.InternalError = err
		return nil
	}

	return ret
}

func (s *Server) handleTokenRequest(w *Response, r *http.Request) *AuthorizeRequest {
	// create the authorization request
	unescapedUri, err := url.QueryUnescape(r.Form.Get("redirect_uri"))
	if err != nil {
		unescapedUri = ""
	}
	ret := &AuthorizeRequest{
		Type:        TOKEN,
		State:       r.Form.Get("state"),
		Scope:       r.Form.Get("scope"),
		RedirectUri: unescapedUri,
		Authorized:  false,
		// this type will generate a token directly, use access token expiration instead.
		Expiration:  s.Config.AccessExpiration,
		HttpRequest: r,
	}

	// must have a valid client
	ret.Client, err = w.Storage.GetClient(r.Form.Get("client_id"))
	if err != nil {
		w.SetErrorState(E_SERVER_ERROR, "", ret.State)
		w.InternalError = err
		return nil
	}
	if ret.Client == nil {
		w.SetErrorState(E_UNAUTHORIZED_CLIENT, "", ret.State)
		return nil
	}
	if ret.Client.GetRedirectUri() == "" {
		w.SetErrorState(E_UNAUTHORIZED_CLIENT, "", ret.State)
		return nil
	}

	// force redirect response to client redirecturl first
	w.SetRedirect(FirstUri(ret.Client.GetRedirectUri(), s.Config.RedirectUriSeparator))

	// check redirect uri
	if ret.RedirectUri == "" {
		ret.RedirectUri = FirstUri(ret.Client.GetRedirectUri(), s.Config.RedirectUriSeparator)
	}
	if err = ValidateUriList(ret.Client.GetRedirectUri(), ret.RedirectUri, s.Config.RedirectUriSeparator); err != nil {
		w.SetErrorState(E_INVALID_REQUEST, "", ret.State)
		w.InternalError = err
		return nil
	}

	return ret
}

func (s *Server) FinishAuthorizeRequest(w *Response, r *http.Request, ar *AuthorizeRequest) {
	// don't process if is already an error
	if w.IsError {
		return
	}

	// force redirect response
	w.SetRedirect(ar.RedirectUri)

	if ar.Authorized {
		if ar.Type == TOKEN {
			w.SetRedirectFragment(true)

			// generate token directly
			ret := &AccessRequest{
				Type:            IMPLICIT,
				Code:            "",
				Client:          ar.Client,
				RedirectUri:     ar.RedirectUri,
				Scope:           ar.Scope,
				GenerateRefresh: false, // per the RFC, should NOT generate a refresh token in this case
				Authorized:      true,
				Expiration:      ar.Expiration,
				UserData:        ar.UserData,
			}

			s.FinishAccessRequest(w, r, ret)
		} else {
			// generate authorization token
			ret := &AuthorizeData{
				Client:      ar.Client,
				CreatedAt:   time.Now(),
				ExpiresIn:   ar.Expiration,
				RedirectUri: ar.RedirectUri,
				State:       ar.State,
				Scope:       ar.Scope,
				UserData:    ar.UserData,
			}

			// generate token code
			code, err := s.AuthorizeTokenGen.GenerateAuthorizeToken(ret)
			if err != nil {
				w.SetErrorState(E_SERVER_ERROR, "", ar.State)
				w.InternalError = err
				return
			}
			ret.Code = code

			// save authorization token
			if err = w.Storage.SaveAuthorize(ret); err != nil {
				w.SetErrorState(E_SERVER_ERROR, "", ar.State)
				w.InternalError = err
				return
			}

			// redirect with code
			w.Output["code"] = ret.Code
			w.Output["state"] = ret.State
		}
	} else {
		// redirect with error
		w.SetErrorState(E_ACCESS_DENIED, "", ar.State)
	}
}
