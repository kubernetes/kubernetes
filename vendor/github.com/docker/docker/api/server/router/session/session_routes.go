package session

import (
	"net/http"

	"golang.org/x/net/context"
)

type invalidRequest struct {
	cause error
}

func (e invalidRequest) Error() string {
	return e.cause.Error()
}

func (e invalidRequest) Cause() error {
	return e.cause
}

func (e invalidRequest) InvalidParameter() {}

func (sr *sessionRouter) startSession(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	err := sr.backend.HandleHTTPRequest(ctx, w, r)
	if err != nil {
		return invalidRequest{err}
	}
	return nil
}
