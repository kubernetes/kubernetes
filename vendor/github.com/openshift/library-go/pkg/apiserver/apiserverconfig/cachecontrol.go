package apiserverconfig

import (
	"net/http"
	"strings"
)

// cacheExcludedPaths is small and simple until the handlers include the cache headers they need
var cacheExcludedPathPrefixes = []string{
	"/swagger-2.0.0.json",
	"/swagger-2.0.0.pb-v1",
	"/swagger-2.0.0.pb-v1.gz",
	"/swagger.json",
	"/swaggerapi",
	"/openapi/",
}

// cacheControlFilter sets the Cache-Control header to the specified value.
func WithCacheControl(handler http.Handler, value string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if _, ok := w.Header()["Cache-Control"]; ok {
			handler.ServeHTTP(w, req)
			return
		}
		for _, prefix := range cacheExcludedPathPrefixes {
			if strings.HasPrefix(req.URL.Path, prefix) {
				handler.ServeHTTP(w, req)
				return
			}
		}

		w.Header().Set("Cache-Control", value)
		handler.ServeHTTP(w, req)
	})
}
