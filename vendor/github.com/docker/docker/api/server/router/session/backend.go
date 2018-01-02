package session

import (
	"net/http"

	"golang.org/x/net/context"
)

// Backend abstracts an session receiver from an http request.
type Backend interface {
	HandleHTTPRequest(ctx context.Context, w http.ResponseWriter, r *http.Request) error
}
