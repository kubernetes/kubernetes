package filters

import (
	"net/http"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/endpoints/request"
	versioninfo "k8s.io/component-base/version"
	"k8s.io/kubernetes/openshift-kube-apiserver/filters/apirequestcount"
)

// WithAPIRequestCountLogging adds a handler that logs counts of api requests.
func WithAPIRequestCountLogging(handler http.Handler, requestLogger apirequestcount.APIRequestLogger) http.Handler {
	currentMinor := version.MustParseSemantic(versioninfo.Get().GitVersion).Minor()
	handlerFunc := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		defer handler.ServeHTTP(w, req)
		info, ok := request.RequestInfoFrom(req.Context())
		if !ok || !info.IsResourceRequest {
			return
		}
		gvr := schema.GroupVersionResource{
			Group:    info.APIGroup,
			Version:  info.APIVersion,
			Resource: info.Resource,
		}
		if minor, ok := apirequestcount.DeprecatedAPIRemovedRelease[gvr]; !ok || minor <= currentMinor {
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
			gvr,
			timestamp,
			user.GetName(),
			req.UserAgent(),
			info.Verb,
		)
	})
	return handlerFunc
}
