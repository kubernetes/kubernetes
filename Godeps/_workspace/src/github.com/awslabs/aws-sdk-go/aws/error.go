package aws

// An APIError is an error returned by an AWS API.
type APIError struct {
	StatusCode int // HTTP status code e.g. 200
	Code       string
	Message    string
	RequestID  string
}

// Error returns the error as a string. Satisfies error interface.
func (e APIError) Error() string {
	return e.Code + ": " + e.Message
}

// Error returns an APIError pointer if the error e is an APIError type.
// If the error is not an APIError nil will be returned.
func Error(e error) *APIError {
	if err, ok := e.(*APIError); ok {
		return err
	} else if err, ok := e.(APIError); ok {
		return &err
	} else {
		return nil
	}
}
