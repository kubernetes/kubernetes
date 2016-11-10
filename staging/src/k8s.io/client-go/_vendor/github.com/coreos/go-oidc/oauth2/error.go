package oauth2

const (
	ErrorAccessDenied            = "access_denied"
	ErrorInvalidClient           = "invalid_client"
	ErrorInvalidGrant            = "invalid_grant"
	ErrorInvalidRequest          = "invalid_request"
	ErrorServerError             = "server_error"
	ErrorUnauthorizedClient      = "unauthorized_client"
	ErrorUnsupportedGrantType    = "unsupported_grant_type"
	ErrorUnsupportedResponseType = "unsupported_response_type"
)

type Error struct {
	Type        string `json:"error"`
	Description string `json:"error_description,omitempty"`
	State       string `json:"state,omitempty"`
}

func (e *Error) Error() string {
	if e.Description != "" {
		return e.Type + ": " + e.Description
	}
	return e.Type
}

func NewError(typ string) *Error {
	return &Error{Type: typ}
}
