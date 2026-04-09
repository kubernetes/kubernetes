package filters

import (
	"net/http"

	apirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// This rejects resource requests whose resourec name exceeds maxLen.
// // This prevents big user given names from reaching request
// paths, like metrics/error handling.
func WithResourceNameLengthLimit(handler http.Handler, maxLen int) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		info, ok := apirequest.RequestInfoFrom(req.Context())
		if !ok || info == nil || !info.IsResourceRequest {
			handler.ServeHTTP(w, req)
			return
		}

		if len(info.Name) > maxLen {
			http.Error(w, "resource name too long", http.StatusBadRequest)
			return
		}

		handler.ServeHTTP(w, req)
	})
}
