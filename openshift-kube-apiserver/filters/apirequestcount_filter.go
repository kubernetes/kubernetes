package filters

import (
	"net/http"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/openshift-kube-apiserver/filters/apirequestcount"
)

// WithAPIRequestCountLogging adds a handler that logs counts of api requests.
func WithAPIRequestCountLogging(handler http.Handler, requestLogger apirequestcount.APIRequestLogger) http.Handler {
	handlerFunc := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer handler.ServeHTTP(w, req)
		info, ok := request.RequestInfoFrom(req.Context())
		if !ok || !info.IsResourceRequest {
			return
		}
		timestamp, ok := request.ReceivedTimestampFrom(req.Context())
		if !ok {
			return
		}
		user, ok := request.UserFrom(req.Context())
		if !ok {
			return
		}
		requestLogger.LogRequest(
			schema.GroupVersionResource{
				Group:    info.APIGroup,
				Version:  info.APIVersion,
				Resource: info.Resource,
			},
			timestamp,
			user.GetName(),
			req.UserAgent(),
			info.Verb,
		)
	})
	return handlerFunc
}
