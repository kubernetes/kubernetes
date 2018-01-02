package debug

import (
	"net/http"
	"net/http/pprof"

	"golang.org/x/net/context"
)

func handlePprof(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	pprof.Handler(vars["name"]).ServeHTTP(w, r)
	return nil
}
