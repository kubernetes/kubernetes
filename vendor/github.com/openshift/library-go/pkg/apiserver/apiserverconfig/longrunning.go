package apiserverconfig

import (
	"net/http"
	"regexp"

	"k8s.io/apimachinery/pkg/util/sets"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfilters "k8s.io/apiserver/pkg/server/filters"
)

// request paths that match this regular expression will be treated as long running
// and not subjected to the default server timeout.
const originLongRunningEndpointsRE = "(/|^)(buildconfigs/.*/instantiatebinary|imagestreamimports)$"

var (
	originLongRunningRequestRE = regexp.MustCompile(originLongRunningEndpointsRE)
	kubeLongRunningFunc        = genericfilters.BasicLongRunningRequestCheck(
		sets.NewString("watch", "proxy"),
		sets.NewString("attach", "exec", "proxy", "log", "portforward"),
	)
)

func IsLongRunningRequest(r *http.Request, requestInfo *apirequest.RequestInfo) bool {
	return originLongRunningRequestRE.MatchString(r.URL.Path) || kubeLongRunningFunc(r, requestInfo)
}
