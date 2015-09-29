package oauth2

import (
	"encoding/json"
	"fmt"
)

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
	Type  string `json:"error"`
	State string `json:"state,omitempty"`
}

func (e *Error) Error() string {
	return e.Type
}

func NewError(typ string) *Error {
	return &Error{Type: typ}
}

func unmarshalError(b []byte) error {
	var oerr Error
	err := json.Unmarshal(b, &oerr)
	if err != nil {
		return fmt.Errorf("unrecognized error: %s", string(b))
	}
	return &oerr
}
