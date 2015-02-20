package osincli

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

// OAuth2 error base
type Error struct {
	Id          string
	Description string
	URI         string
	State       string
}

func (e *Error) Error() string {
	return e.Description
}

func NewError(id, description, uri, state string) *Error {
	return &Error{
		Id:          id,
		Description: description,
		URI:         uri,
		State:       state,
	}
}
