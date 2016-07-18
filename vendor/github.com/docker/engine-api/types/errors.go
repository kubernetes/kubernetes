package types

// ErrorResponse is the response body of API errors.
type ErrorResponse struct {
	Message string `json:"message"`
}
