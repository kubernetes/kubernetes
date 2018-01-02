package middleware

import (
	"net/http"

	"golang.org/x/net/context"
)

// Middleware is an interface to allow the use of ordinary functions as Docker API filters.
// Any struct that has the appropriate signature can be registered as a middleware.
type Middleware interface {
	WrapHandler(func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error) func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error
}
