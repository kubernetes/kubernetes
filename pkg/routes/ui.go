/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package routes

import (
	"net/http"

	"github.com/emicklei/go-restful"
)

const dashboardPath = "/api/v1/proxy/namespaces/kube-system/services/kubernetes-dashboard"

// UIRediect redirects /ui to the kube-ui proxy path.
func UIRedirect() *restful.WebService {
	ws := new(restful.WebService)
	ws.Path("/ui/")
	ws.Doc("redirect to the dashboard")
	ws.Route(ws.GET("/").To(func(req *restful.Request, resp *restful.Response) {
		http.Redirect(resp.ResponseWriter, req.Request, dashboardPath, http.StatusTemporaryRedirect)
	}))
	return ws
}
