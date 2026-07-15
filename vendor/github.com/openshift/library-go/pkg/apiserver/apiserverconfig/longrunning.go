package apiserverconfig

import (
	"net/http"

	"k8s.io/apimachinery/pkg/util/sets"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfilters "k8s.io/apiserver/pkg/server/filters"

	buildv1 "github.com/openshift/api/build/v1"
	imagev1 "github.com/openshift/api/image/v1"
)

var (
	longRunningVerbs        = sets.NewString("watch", "proxy")
	longRunningSubresources = sets.NewString("attach", "exec", "proxy", "log", "portforward")
	kubeLongRunningFunc     = genericfilters.BasicLongRunningRequestCheck(longRunningVerbs, longRunningSubresources)
)

func IsLongRunningRequest(r *http.Request, requestInfo *apirequest.RequestInfo) bool {
	if requestInfo == nil {
		return false
	}

	if requestInfo.APIGroup == buildv1.GroupName &&
		requestInfo.Resource == "buildconfigs" &&
		requestInfo.Subresource == "instantiatebinary" {
		return true
	}
	if requestInfo.APIGroup == imagev1.GroupName &&
		requestInfo.Resource == "imagestreamimports" {
		return true
	}
	if kubeLongRunningFunc(r, requestInfo) {
		return true
	}
	return false
}
