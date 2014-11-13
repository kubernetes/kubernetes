package osin

type DefaultErrorId string

const (
	E_INVALID_REQUEST           string = "invalid_request"
	E_UNAUTHORIZED_CLIENT              = "unauthorized_client"
	E_ACCESS_DENIED                    = "access_denied"
	E_UNSUPPORTED_RESPONSE_TYPE        = "unsupported_response_type"
	E_INVALID_SCOPE                    = "invalid_scope"
	E_SERVER_ERROR                     = "server_error"
	E_TEMPORARILY_UNAVAILABLE          = "temporarily_unavailable"
	E_UNSUPPORTED_GRANT_TYPE           = "unsupported_grant_type"
	E_INVALID_GRANT                    = "invalid_grant"
	E_INVALID_CLIENT                   = "invalid_client"
)

var (
	deferror *DefaultErrors = NewDefaultErrors()
)

// Default errors and messages
type DefaultErrors struct {
	errormap map[string]string
}

// NewDefaultErrors initializes OAuth2 error codes and descriptions.
// http://tools.ietf.org/html/rfc6749#section-4.1.2.1
// http://tools.ietf.org/html/rfc6749#section-4.2.2.1
// http://tools.ietf.org/html/rfc6749#section-5.2
// http://tools.ietf.org/html/rfc6749#section-7.2
func NewDefaultErrors() *DefaultErrors {
	r := &DefaultErrors{errormap: make(map[string]string)}
	r.errormap[E_INVALID_REQUEST] = "The request is missing a required parameter, includes an invalid parameter value, includes a parameter more than once, or is otherwise malformed."
	r.errormap[E_UNAUTHORIZED_CLIENT] = "The client is not authorized to request a token using this method."
	r.errormap[E_ACCESS_DENIED] = "The resource owner or authorization server denied the request."
	r.errormap[E_UNSUPPORTED_RESPONSE_TYPE] = "The authorization server does not support obtaining a token using this method."
	r.errormap[E_INVALID_SCOPE] = "The requested scope is invalid, unknown, or malformed."
	r.errormap[E_SERVER_ERROR] = "The authorization server encountered an unexpected condition that prevented it from fulfilling the request."
	r.errormap[E_TEMPORARILY_UNAVAILABLE] = "The authorization server is currently unable to handle the request due to a temporary overloading or maintenance of the server."
	r.errormap[E_UNSUPPORTED_GRANT_TYPE] = "The authorization grant type is not supported by the authorization server."
	r.errormap[E_INVALID_GRANT] = "The provided authorization grant (e.g., authorization code, resource owner credentials) or refresh token is invalid, expired, revoked, does not match the redirection URI used in the authorization request, or was issued to another client."
	r.errormap[E_INVALID_CLIENT] = "Client authentication failed (e.g., unknown client, no client authentication included, or unsupported authentication method)."
	return r
}

func (e *DefaultErrors) Get(id string) string {
	if m, ok := e.errormap[id]; ok {
		return m
	}
	return id
}
