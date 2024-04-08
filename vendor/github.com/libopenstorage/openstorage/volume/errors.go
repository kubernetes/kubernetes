package volume

// CredentialError error returned for credential operations
type CredentialError struct {
	// Msg is human understandable error message
	Msg string
}

func NewCredentialError(msg string) error {
	return &CredentialError{Msg: msg}
}

func (e *CredentialError) Error() string {
	return e.Msg
}
